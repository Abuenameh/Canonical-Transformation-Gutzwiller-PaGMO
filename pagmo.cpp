#include <pagmo/src/pagmo.h>

#include "gutzwiller.hpp"

using namespace std;
using namespace pagmo;
using namespace pagmo::problem;

//static inline double Sqr(double x) {
//    return x * x;
//}
//
//class energy : public base {
//public:
//    energy(int n, vector<double>& U_, vector<double>& J_, double mu_, double theta_) : base(n), U(U_), J(J_), mu(mu_), theta(theta_) {
//        costh = cos(theta);
//        sinth = sin(theta);
//        cos2th = cos(2*theta);
//        sin2th = sin(2*theta);
//    }
//    
//    base_ptr clone() const;
//    void objfun_impl(fitness_vector& f, const decision_vector& x) const;
//    
//private:
//    vector<double>& U;
//    vector<double>& J;
//    double mu;
//    double theta;
//    double costh;
//    double sinth;
//    double cos2th;
//    double sin2th;
//};

base_ptr energy::clone() const {
    return base_ptr(new energy(*this));
}

void energy::objfun_impl(fitness_vector& En, const decision_vector& x) const {

    doublecomplex expth = exp(doublecomplex(0, 1) * theta);
    doublecomplex expmth = exp(-doublecomplex(0, 1) * theta);
    doublecomplex exp2th = exp(doublecomplex(0, 1)*2.0 * theta);
    doublecomplex expm2th = exp(-doublecomplex(0, 1)*2.0 * theta);

    doublecomplex Ec = 0;

    const doublecomplex * f[L];
    vector<double> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<const doublecomplex*> (&x[2 * i * dim]);
        for (int n = 0; n <= nmax; n++) {
            norm2[i] += norm(f[i][n]);
        }
    }

    vector<doublecomplex> E0s(L, 0), E1j1s(L, 0), E1j2s(L, 0),
            E2j1s(L, 0), E2j2s(L, 0), E3j1s(L, 0), E3j2s(L, 0),
            E4j1j2s(L, 0), E4j1k1s(L, 0), E4j2k2s(L, 0),
            E5j1j2s(L, 0), E5j1k1s(L, 0), E5j2k2s(L, 0);


    for (int i = 0; i < L; i++) {

        int k1 = mod(i - 2);
        int j1 = mod(i - 1);
        int j2 = mod(i + 1);
        int k2 = mod(i + 2);

        doublecomplex E0 = 0;
        doublecomplex E1j1 = 0;
        doublecomplex E1j2 = 0;
        doublecomplex E2j1 = 0;
        doublecomplex E2j2 = 0;
        doublecomplex E3j1 = 0;
        doublecomplex E3j2 = 0;
        doublecomplex E4j1j2 = 0;
        doublecomplex E4j1k1 = 0;
        doublecomplex E4j2k2 = 0;
        doublecomplex E5j1j2 = 0;
        doublecomplex E5j1k1 = 0;
        doublecomplex E5j2k2 = 0;

        for (int n = 0; n <= nmax; n++) {
            E0 += (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];

            if (n < nmax) {
                E1j1 += -J[j1] * expth * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                        * f[i][n] * f[j1][n + 1];
                E1j2 += -J[i] * expmth * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
                        * f[j2][n + 1];

                if (n > 0) {
                    E2j1 += 0.5 * J[j1] * J[j1] * exp2th * g(n, n) * g(n - 1, n + 1)
                            * ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1]
                            * (1 / eps(U, i, j1, n, n) - 1 / eps(U, i, j1, n - 1, n + 1));
                    E2j2 += 0.5 * J[i] * J[i] * expm2th * g(n, n) * g(n - 1, n + 1)
                            * ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1]
                            * (1 / eps(U, i, j2, n, n) - 1 / eps(U, i, j2, n - 1, n + 1));
                }

                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1) {
                        E3j1 += 0.5 * (J[j1] * J[j1] / eps(U, i, j1, n, m)) * g(n, m)
                                * g(m - 1, n + 1)
                                * (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1]
                                - ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
                        E3j2 += 0.5 * (J[i] * J[i] / eps(U, i, j2, n, m)) * g(n, m)
                                * g(m - 1, n + 1)
                                * (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1]
                                - ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);
                    }
                }

                if (n > 0) {
                    E4j1j2 += 0.5 * (J[j1] * J[i] / eps(U, i, j1, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[j2][n]
                            * f[i][n - 1] * f[j1][n] * f[j2][n + 1];
                    E4j1j2 += 0.5 * (J[i] * J[j1] / eps(U, i, j2, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[j1][n]
                            * f[i][n - 1] * f[j2][n] * f[j1][n + 1];
                    E4j1k1 += 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n - 1] * ~f[k1][n]
                            * f[i][n] * f[j1][n + 1] * f[k1][n - 1];
                    E4j2k2 += 0.5 * (J[i] * J[j2] / eps(U, i, j2, n, n)) * g(n, n)
                            * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n - 1] * ~f[k2][n]
                            * f[i][n] * f[j2][n + 1] * f[k2][n - 1];
                    E4j1j2 -= 0.5 * (J[j1] * J[i] / eps(U, i, j1, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                            * ~f[j2][n - 1] * f[i][n - 1] * f[j1][n + 1] * f[j2][n];
                    E4j1j2 -= 0.5 * (J[i] * J[j1] / eps(U, i, j2, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n + 1] * ~f[j2][n]
                            * ~f[j1][n - 1] * f[i][n - 1] * f[j2][n + 1] * f[j1][n];
                    E4j1k1 -= 0.5 * (J[j1] * J[k1] / eps(U, i, j1, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j1][n - 1]
                            * ~f[k1][n + 1] * f[i][n - 1] * f[j1][n + 1] * f[k1][n];
                    E4j2k2 -= 0.5 * (J[i] * J[j2] / eps(U, i, j2, n - 1, n + 1))
                            * g(n, n) * g(n - 1, n + 1) * ~f[i][n] * ~f[j2][n - 1]
                            * ~f[k2][n + 1] * f[i][n - 1] * f[j2][n + 1] * f[k2][n];
                }

                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1 && n < nmax) {
                        E5j1j2 += 0.5 * (J[j1] * J[i] * exp2th / eps(U, i, j1, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
                                * ~f[j2][m] * f[i][n + 1] * f[j1][m] * f[j2][m - 1];
                        E5j1j2 += 0.5 * (J[i] * J[j1] * expm2th / eps(U, i, j2, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
                                * ~f[j1][m] * f[i][n + 1] * f[j2][m] * f[j1][m - 1];
                        E5j1k1 += 0.5 * (J[j1] * J[k1] * exp2th / eps(U, i, j1, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m - 1]
                                * ~f[k1][n] * f[i][n] * f[j1][m - 1] * f[k1][n + 1];
                        E5j2k2 += 0.5 * (J[i] * J[j2] * expm2th / eps(U, i, j2, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m - 1]
                                * ~f[k2][n] * f[i][n] * f[j2][m - 1] * f[k2][n + 1];
                        E5j1j2 -= 0.5 * (J[j1] * J[i] * exp2th / eps(U, i, j1, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n] * ~f[j1][m - 1]
                                * ~f[j2][m] * f[i][n] * f[j1][m] * f[j2][m - 1];
                        E5j1j2 -= 0.5 * (J[i] * J[j1] * expm2th / eps(U, i, j2, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n] * ~f[j2][m - 1]
                                * ~f[j1][m] * f[i][n] * f[j2][m] * f[j1][m - 1];
                        E5j1k1 -= 0.5 * (J[j1] * J[k1] * exp2th / eps(U, i, j1, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j1][m]
                                * ~f[k1][n] * f[i][n] * f[j1][m] * f[k1][n + 1];
                        E5j2k2 -= 0.5 * (J[i] * J[j2] * expm2th / eps(U, i, j2, n, m))
                                * g(n, m) * g(m - 1, n + 1) * ~f[i][n + 1] * ~f[j2][m]
                                * ~f[k2][n] * f[i][n] * f[j2][m] * f[k2][n + 1];
                    }
                }
            }

        }

        Ec += E0 / norm2[i];
        
                Ec += E1j1 / (norm2[i] * norm2[j1]);
                Ec += E1j2 / (norm2[i] * norm2[j2]);
                
                Ec += E2j1 / (norm2[i] * norm2[j1]);
                Ec += E2j2 / (norm2[i] * norm2[j2]);
                
                Ec += E3j1 / (norm2[i] * norm2[j1]);
                Ec += E3j2 / (norm2[i] * norm2[j2]);
                
                Ec += E4j1j2 / (norm2[i] * norm2[j1] * norm2[j2]);
                Ec += E4j1k1 / (norm2[i] * norm2[j1] * norm2[k1]);
                Ec += E4j2k2 / (norm2[i] * norm2[j2] * norm2[k2]);
                
                Ec += E5j1j2 / (norm2[i] * norm2[j1] * norm2[j2]);
                Ec += E5j1k1 / (norm2[i] * norm2[j1] * norm2[k1]);
                Ec += E5j2k2 / (norm2[i] * norm2[j2] * norm2[k2]);
    }
    
    En[0] = Ec.real();
    
//#include "vars.cpp"
//#include "return.cpp"
//    
//    En[0] = E;
}