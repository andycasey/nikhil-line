
data {
    int<lower=1> N;
    vector[N] x;
    vector[N] x_err;
    vector[N] y;
    vector[N] y_err;
}

parameters {
    real a;
    real b;
    vector[N] x_t; // true x values
}

model {
    x_t ~ normal(x, x_err);
    y ~ normal(a + b*x_t, y_err);
}
