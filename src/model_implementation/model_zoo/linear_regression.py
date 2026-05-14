from sklearn.linear_model import LinearRegression


def build_linear_regression():
    """Build a Linear Regression model with default sklearn settings.

    Formula: ŷ = β₀ + β₁x₁ + β₂x₂ + ... + β₁₆x₁₆

    Each β (beta) is a weight the model learns — it says "how much does
    this feature increase or decrease the predicted wait time?"

    Pros: Simplest model, fastest to train, fully interpretable.
    Cons: Can only model linear relationships. Cannot capture interactions
          like "Monday + 10am = extra bad."
    """
    return LinearRegression()