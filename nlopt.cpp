#include <vector>
#include <cmath>

using namespace std;

struct funcdata2 {
    vector<double>& U;
    vector<double>& J;
    double mu;
    double theta;
};

static inline double Sqr(double x) {
    return x * x;
}

double energyfunc(unsigned ndim, const vector<double> f, vector<double> grad, void *data) {
    funcdata2* fdata = static_cast<funcdata2*> (data);
    vector<double>& U = fdata->U;
    vector<double>& J = fdata->J;
    double mu = fdata->mu;
    double theta = fdata->theta;
    double costh = cos(theta);
    double sinth = sin(theta);
    double cos2th = cos(2*theta);
    double sin2th = sin(2*theta);

#include "vars.cpp"
#include "return.cpp"
    if(!grad.empty()) {
#include "grad.cpp"
    }
    return E;
}
