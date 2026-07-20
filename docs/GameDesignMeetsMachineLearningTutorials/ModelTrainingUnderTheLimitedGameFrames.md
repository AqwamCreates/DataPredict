# Model Training Under The Limited Game Frames

* Per Frame (Physics / Render) -> Model must be fast. Ideally use single datapoints or online models here.

* Per Interval -> Model calculation time must not exceed the interval. Ideally use mini-batch training here.

* Per Session End -> Batch training is allowed.
