# Long-only DCA / Martingale backtest

Selected parameters: MA=12h, first drop=4.0%, add step=2.0%, multiplier=1.50, take profit=1.0%, stop loss=10.0%, max legs=4.
Validation mean/median/p10 return: -0.16%/1.44%/-3.32%; worst validation return: -10.61%.
Holdout mean/median return: 0.86%/0.88%; profitable holdout symbols: 18/20.
The configuration is selected on validation only; a negative validation score is not hidden by holdout results.
