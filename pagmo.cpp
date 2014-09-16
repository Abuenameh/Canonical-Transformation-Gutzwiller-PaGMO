#include <pagmo/src/pagmo.h>

using namespace std;
using namespace pagmo;
using namespace pagmo::problem;

static inline double Sqr(double x) {
    return x * x;
}

class energy : public base {
public:
    energy(int n, vector<double>& U_, vector<double>& J_, double mu_, double theta_) : base(n), U(U_), J(J_), mu(mu_), theta(theta_) {
        costh = cos(theta);
        sinth = sin(theta);
        cos2th = cos(2*theta);
        sin2th = sin(2*theta);
    }
    
    base_ptr clone() const;
    void objfun_impl(fitness_vector& f, const decision_vector& x) const;
    
private:
    vector<double>& U;
    vector<double>& J;
    double mu;
    double theta;
    double costh;
    double sinth;
    double cos2th;
    double sin2th;
};

base_ptr energy::clone() const {
    return base_ptr(new energy(*this));
}

void energy::objfun_impl(fitness_vector& En, const decision_vector& f) const {

#include "vars.cpp"
#include "return.cpp"
    
    En[0] = E;
}