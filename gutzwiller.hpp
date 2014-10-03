/* 
 * File:   gutzwiller.hpp
 * Author: Abuenameh
 *
 * Created on August 10, 2014, 10:45 PM
 */

#ifndef GUTZWILLER_HPP
#define	GUTZWILLER_HPP

#include <complex>
#include <vector>
#include <iostream>


using namespace std;

#include <pagmo/src/pagmo.h>

using namespace pagmo;
using namespace pagmo::problem;

//#include <bayesopt/bayesopt.hpp>

using namespace boost::numeric;

typedef complex<double> doublecomplex;

#define L 5
#define nmax 5
#define idim (nmax+1)
#define dim (nmax+1)

template<class T>
complex<T> operator~(const complex<T> a) {
	return conj(a);
}

//struct funcdata {
//    bool canonical;
//	vector<double> U;
//	double mu;
//	vector<double> J;
//    double theta;
//    double Emin;
//    vector<double> xmin;
//    vector<double> x0;
////	double* U;
////	double mu;
////	double* J;
//};

struct funcdata {
    bool canonical;
//	vector<double> U;
	double* U;
	double mu;
//	vector<double> J;
	double* J;
    double theta;
    double Emin;
//    vector<double> xmin;
    double* xmin;
//    vector<double> x0;
    double* x0;
//	double* U;
//	double mu;
//	double* J;
};

struct funcdata2 {
    vector<double>& U;
    vector<double>& J;
    double mu;
    double theta;
    vector<double> Ei;
};

struct phase_parameters {
    double theta;
    bool canonical;
};

struct device_parameters {
	double* U;
	double mu;
	double* J;
    double theta;
};

inline int mod(int i) {
	return (i + L) % L;
}

inline double g(int n, int m) {
	return sqrt(1.0*(n + 1) * m);
}

extern vector<double> nu;

inline double eps(vector<double> U, int i, int j, int n, int m) {
	return n * U[i] - (m - 1) * U[j];
}

//double Encfunc(unsigned ndim, const double *x, double *grad, void *data);
//double Ecfunc(unsigned ndim, const double *x, double *grad, void *data);
//double Ecfunc(int i, unsigned ndim, const double *x, double *grad, void *data);
//double Ecfuncp(ProblemSetup *setup, double *x);
//
//double Ecfunci(unsigned ndim, int i, const double *x, double *grad, void *data);
//double Ecfunci(unsigned ndim, int i, int which, const double *x, double *grad, void *data);

double Efunc(unsigned ndim, const double *f, double *grad, void *data);

double energyfunc(const vector<double>& x, vector<double>& grad, void *data);

//double energy(double *f, int ndim, void *data);

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

/*class bayesfunc : public bayesopt::ContinuousModel {
private:
    funcdata2& data;

public:

    bayesfunc(bopt_params parm, funcdata2& data_) : ContinuousModel(2 * L*dim, parm), data(data_) {
        vectord lb = svectord(2*L*dim, -1);
        vectord ub = svectord(2*L*dim, 1);
        setBoundingBox(lb, ub);
    }

    virtual double evaluateSample(const vectord& x);
    };*/
    
#endif	/* GUTZWILLER_HPP */

