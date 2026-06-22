Modern risk management and portfolio analysis rely heavily on the ability to simulate plausible future market scenarios. Rather than depending solely on the limited window of observed history, practitioners seek to generate synthetic financial time series that are statistically consistent with historical data while providing a richer and more diverse set of forward-looking scenarios. This is particularly critical for applications such as Value-at-Risk (VaR) estimation, stress testing, asset-liability management, and Monte Carlo-based portfolio optimization. However, financial time series are not well-behaved stochastic processes. Unlike the idealized assumptions underpinning classical models — independent and identically distributed returns, Gaussian innovations, constant volatility — real market data exhibits a rich and complex set of statistical properties that any credible simulation framework must reproduce.

Overall, financial time series show four heuristics / stylized facts that consistently appear across asset classes, geographies, and time periods:

1.	Non-linear dependencies: While raw returns are largely uncorrelated over time, their squares and absolute values exhibit significant and persistent autocorrelation. This reflects the fact that large moves tend to cluster together, a phenomenon that linear models entirely miss. Standard tests such as the Ljung-Box test on squared returns routinely reject the null hypothesis of independence, confirming that the dependence structure of financial returns is fundamentally non-linear.
2.	Time-varying correlations and regime changes: The statistical relationships between assets are not stable. Correlations shift dramatically across market regimes: during calm periods, assets may behave quasi-independently, while during crises, correlations spike sharply as contagion and flight-to-quality dynamics dominate. A synthetic scenario generator that assumes fixed correlations will systematically underestimate joint tail risk precisely when it matters most.
3.	Heteroskedasticity and volatility clustering: Financial volatility is not constant. Periods of high volatility tend to be followed by further high volatility, and calm periods tend to persist. This clustering effect means that the conditional variance of returns is time-varying and path-dependent. Failing to reproduce this property leads to scenarios that are either uniformly too calm or uniformly too stressed, neither of which reflects realistic market dynamics.
4.	Fat tails and asymmetric dependence: The distribution of financial returns has heavier tails than the Gaussian distribution predicts. Extreme losses occur far more frequently than a normal model would suggest, and the dependence structure is asymmetric: assets tend to co-crash more strongly than they co-rally. This asymmetry, captured by concepts such as lower tail dependence in copula theory, is a defining feature of financial risk that Gaussian-based models structurally cannot capture.

Generating synthetic time series that jointly satisfy all four heuristics is a challenge:

1.	Trade-off between flexibility and tractability: Parametric models (multivariate GARCH or factor models with t-distributed innovations) offer analytical tractability and the ability to extrapolate beyond observed history, but they impose strong structural assumptions that may not hold in practice. Non-parametric approaches such as bootstrap resampling preserve the empirical distribution exactly but are bounded by historical experience and cannot generate scenarios outside the observed support.
2.	Independence: all four heuristics are not independent of one another. Volatility clustering and fat tails are deeply intertwined: the heavy tails of the unconditional return distribution arise in large part from the mixing of different volatility regimes. Similarly, time-varying correlations and non-linear dependencies are two manifestations of the same underlying regime-switching dynamics. A model that targets one property in isolation may inadvertently distort another.
3.	Curse of dimensionality: it compounds these difficulties in a multivariate setting. Reproducing realistic marginal distributions for each asset is already non-trivial; ensuring that the joint dependence structure (among others, tail dependence, dynamic correlations, and cross-asset contagion) is also realistic adds a further layer of complexity that grows rapidly with the number of assets.
4.	Model risk: Any single simulation methodology embeds assumptions that may be misspecified. Relying on a single approach exposes the analysis to the blind spots of that model, which may be precisely the scenarios most relevant for risk management.

**YOUR TASK**

**Build an economic scenario generator that produces 10 plausible joint return scenarios across all 17 asset classes.**

The attached dataset contains 17 assets representing various asset classes and geographical areas. A scenario is defined here as a single-period vector of returns drawn simultaneously for all 17 assets. "Plausible" means statistically consistent with the properties of the historical data, as described by the four stylized facts above. Your generator must satisfy the following requirements:

**1. Respect the four stylized facts**
Your generated scenarios must be consistent with the four properties described above. For each property, implement a specific, explicit control or diagnostic that demonstrates your generated data complies with it. Simply choosing a model that theoretically accounts for these properties is not sufficient — you must show empirically that your output does.

**2. Preserve the full dependence structure**
Scenarios must always be generated jointly across all 17 assets. Never generate returns for one asset independently of the remaining 16. The correlation and tail-dependence structure observed in the historical data must be reflected in your output.

**3. Justify every modelling choice**
You are free to use any technique you consider appropriate — GARCH-based models, copulas, GANs, VAEs, or any combination thereof. However, you must be able to explain and defend every step of your approach: why you chose a given model, what assumptions it makes, and how you verified that those assumptions are met.

**4. Time periods**
Scenarios must be generated daily, from 2010-01-01 until 2025-06-25

Submit your code, your 10 generated scenarios (as a table of 17 returns per scenario), and a written explanation of your methodology and validation approach.