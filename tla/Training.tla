---- MODULE Training ----
EXTENDS Integers, Sequences

CONSTANTS Workers, MaxVersion, BatchSize, MaxQueue, MaxCrashes

VARIABLES
    redisLatestModel,
    redisOpponentPool,
    redisRolloutQueue,
    workerStep,
    workerModel,
    workerOpponent,
    learnerStep,
    learnerModel,
    learnerBatch,
    crashCount

redisVars == <<redisLatestModel, redisOpponentPool, redisRolloutQueue>>
workerVars == <<workerStep, workerModel, workerOpponent>>
learnerVars == <<learnerStep, learnerModel, learnerBatch>>
vars == <<redisVars, workerVars, learnerVars, crashCount>>

NoModel == [run |-> -1, version |-> -1]
FirstModel(run) == [run |-> run, version |-> 0]

Init ==
    /\ redisLatestModel = FirstModel(0)
    /\ redisOpponentPool = {FirstModel(0)}
    /\ redisRolloutQueue = <<>>
    /\ workerStep = [w \in Workers |-> "fetchModel"]
    /\ workerModel = [w \in Workers |-> NoModel]
    /\ workerOpponent = [w \in Workers |-> NoModel]
    /\ learnerStep = "collectRollouts"
    /\ learnerModel = FirstModel(0)
    /\ learnerBatch = <<>>
    /\ crashCount = 0

FetchModel(w) ==
    /\ workerStep[w] = "fetchModel"
    /\ workerModel' = [workerModel EXCEPT ![w] = redisLatestModel]
    /\ workerStep' = [workerStep EXCEPT ![w] = "pickOpponent"]
    /\ UNCHANGED <<workerOpponent>>
    /\ UNCHANGED redisVars
    /\ UNCHANGED learnerVars
    /\ UNCHANGED crashCount

PickOpponent(w) ==
    /\ workerStep[w] = "pickOpponent"
    /\ workerModel[w] \in redisOpponentPool
    /\ \E opponent \in redisOpponentPool :
        workerOpponent' = [workerOpponent EXCEPT ![w] = opponent]
    /\ workerStep' = [workerStep EXCEPT ![w] = "playAndPush"]
    /\ UNCHANGED <<workerModel>>
    /\ UNCHANGED redisVars
    /\ UNCHANGED learnerVars
    /\ UNCHANGED crashCount

AbandonDeadRun(w) ==
    /\ workerStep[w] = "pickOpponent"
    /\ workerModel[w] \notin redisOpponentPool
    /\ workerStep' = [workerStep EXCEPT ![w] = "fetchModel"]
    /\ UNCHANGED <<workerModel, workerOpponent>>
    /\ UNCHANGED redisVars
    /\ UNCHANGED learnerVars
    /\ UNCHANGED crashCount

PlayAndPush(w) ==
    /\ workerStep[w] = "playAndPush"
    /\ LET pushed == Append(redisRolloutQueue,
                            [model |-> workerModel[w], opponent |-> workerOpponent[w]])
       IN redisRolloutQueue' = IF Len(pushed) > MaxQueue THEN Tail(pushed) ELSE pushed
    /\ workerStep' = [workerStep EXCEPT ![w] = "fetchModel"]
    /\ UNCHANGED <<workerModel, workerOpponent>>
    /\ UNCHANGED <<redisLatestModel, redisOpponentPool>>
    /\ UNCHANGED learnerVars
    /\ UNCHANGED crashCount

PopRollout ==
    /\ learnerStep = "collectRollouts"
    /\ redisRolloutQueue # <<>>
    /\ LET rollout == Head(redisRolloutQueue)
           isStale == rollout.model # learnerModel
           newBatch == IF isStale THEN learnerBatch ELSE Append(learnerBatch, rollout)
       IN /\ learnerBatch' = newBatch
          /\ learnerStep' = IF Len(newBatch) >= BatchSize THEN "trainAndPublish" ELSE "collectRollouts"
    /\ redisRolloutQueue' = Tail(redisRolloutQueue)
    /\ UNCHANGED <<redisLatestModel, redisOpponentPool, learnerModel>>
    /\ UNCHANGED workerVars
    /\ UNCHANGED crashCount

TrainAndPublish ==
    /\ learnerStep = "trainAndPublish"
    /\ learnerModel.version < MaxVersion
    /\ LET trained == [learnerModel EXCEPT !.version = @ + 1]
       IN /\ learnerModel' = trained
          /\ redisLatestModel' = trained
          /\ redisOpponentPool' = redisOpponentPool \union {trained}
    /\ learnerBatch' = <<>>
    /\ learnerStep' = "collectRollouts"
    /\ UNCHANGED redisRolloutQueue
    /\ UNCHANGED workerVars
    /\ UNCHANGED crashCount

WorkerCrashAndRestart(w) ==
    /\ crashCount < MaxCrashes
    /\ crashCount' = crashCount + 1
    /\ workerStep' = [workerStep EXCEPT ![w] = "fetchModel"]
    /\ workerModel' = [workerModel EXCEPT ![w] = NoModel]
    /\ workerOpponent' = [workerOpponent EXCEPT ![w] = NoModel]
    /\ UNCHANGED redisVars
    /\ UNCHANGED learnerVars

LearnerCrashAndStartNewRun ==
    /\ crashCount < MaxCrashes
    /\ crashCount' = crashCount + 1
    /\ LET fresh == FirstModel(learnerModel.run + 1)
       IN /\ learnerModel' = fresh
          /\ redisLatestModel' = fresh
          /\ redisOpponentPool' = {fresh}
    /\ redisRolloutQueue' = <<>>
    /\ learnerBatch' = <<>>
    /\ learnerStep' = "collectRollouts"
    /\ UNCHANGED workerVars

Next ==
    \/ \E w \in Workers : FetchModel(w) \/ PickOpponent(w) \/ AbandonDeadRun(w) \/ PlayAndPush(w) \/ WorkerCrashAndRestart(w)
    \/ PopRollout
    \/ TrainAndPublish
    \/ LearnerCrashAndStartNewRun

Fairness ==
    /\ \A w \in Workers : WF_vars(FetchModel(w) \/ PickOpponent(w) \/ AbandonDeadRun(w) \/ PlayAndPush(w))
    /\ WF_vars(PopRollout)
    /\ WF_vars(TrainAndPublish)

Spec == Init /\ [][Next]_vars /\ Fairness

TrainsOnlyFreshRollouts ==
    \A i \in 1..Len(learnerBatch) : learnerBatch[i].model = learnerModel

WorkersNeverAheadOfLearner ==
    \A w \in Workers :
        \/ workerModel[w].run < learnerModel.run
        \/ /\ workerModel[w].run = learnerModel.run
           /\ workerModel[w].version <= learnerModel.version

EventuallyReachesMaxVersion == <>(learnerModel.version = MaxVersion)

====
