/*
 * File:   main.cpp
 * Author: Abuenameh
 *
 * Created on August 6, 2014, 11:21 PM
 */

#include <cstdlib>
#include <cstdio>
#include <math.h>
#include <nlopt.h>
#include <complex>
#include <iostream>
#include <queue>
//#include <thread>
#include <nlopt.hpp>

#include <boost/array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/date_time.hpp>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/progress.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>

#include <pagmo/src/pagmo.h>

//#include <bayesopt/bayesopt.hpp>

#include "mathematica.hpp"
#include "gutzwiller.hpp"

using namespace std;

//using boost::lexical_cast;
using namespace boost;
using namespace boost::random;
using namespace boost::filesystem;
using namespace boost::posix_time;

using namespace pagmo;
using namespace pagmo::problem;
using namespace pagmo::algorithm;

typedef boost::array<double, L> Parameter;

//template<typename T> void printMath(ostream& out, string name, T& t) {
//    out << name << "=" << ::math(t) << ";" << endl;
//}
//
//template<typename T> void printMath(ostream& out, string name, int i, T& t) {
//    out << name << "[" << i << "]" << "=" << ::math(t) << ";" << endl;
//}

double M = 1000;
double g13 = 2.5e9;
double g24 = 2.5e9;
double delta = 1.0e12;
double Delta = -2.0e10;
double alpha = 1.1e7;

double Ng = sqrt(M) * g13;

double JW(double W) {
    return alpha * (W * W) / (Ng * Ng + W * W);
}

double JWij(double Wi, double Wj) {
    return alpha * (Wi * Wj) / (sqrt(Ng * Ng + Wi * Wi) * sqrt(Ng * Ng + Wj * Wj));
}

Parameter JW(Parameter W) {
    Parameter v;
    for (int i = 0; i < L; i++) {
        v[i] = W[i] / sqrt(Ng * Ng + W[i] * W[i]);
    }
    Parameter J;
    for (int i = 0; i < L - 1; i++) {
        J[i] = alpha * v[i] * v[i + 1];
    }
    J[L - 1] = alpha * v[L - 1] * v[0];
    return J;
}

double UW(double W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

Parameter UW(Parameter W) {
    Parameter U;
    for (int i = 0; i < L; i++) {
        U[i] = -2 * (g24 * g24) / Delta * (Ng * Ng * W[i] * W[i]) / ((Ng * Ng + W[i] * W[i]) * (Ng * Ng + W[i] * W[i]));
    }
    return U;
}

boost::mutex progress_mutex;
boost::mutex points_mutex;

struct Point {
    int i;
    int j;
    double x;
    double mu;
};

//double Efunc(unsigned ndim, const double *x, double *grad, void *data) {
//    funcdata* parms = static_cast<funcdata*> (data);
//    if (parms->canonical) {
//        return Ecfunc(ndim, x, grad, data);
//    } else {
//        return Encfunc(ndim, x, grad, data);
//    }
//}

void norm2s(unsigned m, double *result, unsigned ndim, const double* x,
        double* grad, void* data);

double norm(const vector<double> x, vector<double>& norms) {
    const doublecomplex * f[L];
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<const doublecomplex*> (&x[2 * i * idim]);
    }

    norms.resize(L);

    //    double norm = 1;
    for (int i = 0; i < L; i++) {
        double normi = 0;
        for (int n = 0; n <= nmax; n++) {
            normi += norm(f[i][n]);
        }
        //        norm *= normi;
        norms[i] = sqrt(normi);
    }
    //    return norm;
    return 0;
}

int min(int a, int b) {
    return a < b ? a : b;
}

int max(int a, int b) {
    return a > b ? a : b;
}

struct fresults {
    multi_array<double, 2>& fmin;
    multi_array<vector<double>, 2>& fn0;
    multi_array<vector<double>, 2>& fmax;
    multi_array<vector<double>, 2>& f0;
    multi_array<vector<double>, 2>& fth;
    multi_array<vector<double>, 2>& fth2;
    multi_array<vector<double>, 2>& f2th;
};

struct results {
    multi_array<double, 2 >& E0res;
    multi_array<double, 2 >& Ethres;
    multi_array<double, 2 >& Eth2res;
    multi_array<double, 2 >& E2thres;
    multi_array<double, 2 >& fs;
    multi_array<double, 2 >& res0;
    multi_array<double, 2 >& resth;
    multi_array<double, 2 >& resth2;
    multi_array<double, 2 >& res2th;
};

#define NSAMP 200

void phasepoints(Parameter& xi, phase_parameters pparms, queue<Point>& points, /*multi_array<vector<double>, 2 >& f0*/fresults& fres, /*multi_array<double, 2 >& E0res, multi_array<double, 2 >& Ethres, multi_array<double, 2 >& Eth2res, multi_array<double, 2 >& fs,*/results& results, progress_display& progress) {

    mt19937 rng;
    rng.seed(time(NULL));
    uniform_real_distribution<> uni(-1, 1);

    int ndim = 2 * L * idim;

    vector<double> x(ndim);
    doublecomplex * f[L];
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<doublecomplex*> (&x[2 * i * idim]);
    }

    vector<double> U(L), J(L);

    vector<double> x0(ndim);
    doublecomplex * f0[L];
    for (int i = 0; i < L; i++) {
        f0[i] = reinterpret_cast<doublecomplex*> (&x0[2 * i * idim]);
    }

    vector<double> xabs(ndim / 2);
    double* fabs[L];
    for (int i = 0; i < L; i++) {
        fabs[i] = &xabs[i * idim];
    }

    vector<double> fn0(L);
    vector<double> fmax(L);

    vector<double> norms(L);

    for (int i = 0; i < L; i++) {
        //        U[i] = 1 + 0.2 * uni(rng);
        //        U[i] = 1;
    }

    funcdata data;
    data.canonical = pparms.canonical;
    //        parms.theta = theta;

    double theta = pparms.theta;

        funcdata2 fdata = {U, J, 0, 0, vector<double>(6)};

        double scale = 1;

    for (;;) {
        Point point;
        {
            boost::mutex::scoped_lock lock(points_mutex);
            if (points.empty()) {
                break;
            }
            point = points.front();
            points.pop();
        }
        //        cout << "Got queued" << endl;

        //
        //    vector<double> U(L), J(L);
        double W[L];
        for (int i = 0; i < L; i++) {
            W[i] = xi[i] * point.x;
        }
        for (int i = 0; i < L; i++) {
            //            double Wi = xi[i] * point.x;
            //            double Wj = xi[i] * point.x;

            U[i] = UW(W[i]) / UW(point.x) / scale;
            J[i] = JWij(W[i], W[mod(i + 1)]) / UW(point.x) / scale;
            //            U[i] = 1;
            //            J[i] = JW(point.x)/UW(point.x);

            ////		U[i] = 0.1 * sqrt(i + 1);
            ////		J[i] = 0.1 * min(i + 1, mod(i + 1) + 1)
            ////			+ 0.2 * max(i + 1, mod(i + 1) + 1);
            //            U[i] = 1+0.2*uni(rng);
            //                        J[i] = point.x;
        }
//        cout << ::math(U) << endl << ::math(J) << endl;
        //        {
        //            boost::mutex::scoped_lock lock(points_mutex);
        //            cout << ::math(U) << endl;
        //        }

        //    
        //	parameters parms;
        data.J = J.data();
        data.U = U.data();
        data.mu = point.mu / scale;
        data.Emin = DBL_MAX;
        vector<double> qwe(ndim);
        data.xmin = qwe.data();
//        data.xmin = vector<double>(ndim);
        
        nlopt::opt localopt(nlopt::LD_LBFGS, ndim);
        localopt.set_lower_bounds(-1);
        localopt.set_upper_bounds(1);
        localopt.set_min_objective(energyfunc, &fdata);
//        localopt.set_ftol_rel(1e-12);
//        localopt.set_xtol_rel(1e-12);
//        localopt.set_ftol_abs(1e-12);
//        localopt.set_xtol_abs(1e-12);
        fdata.mu = point.mu/scale;
        
        vector<double> popx;
        
        de_1220 algo(2000);
//        sa_corana algo(10000, 1000, 0.01, 10, 20);
//        jde algo(1000,14);
//        de local(1000);
//        mbh algo(local);
//        mde_pbx algo(10000);
//        sga algo(500);
//        pso algo(500);
        gsl_bfgs2 lalgo(100, 1e-8, 1e-8, 0.01, 1e-10);
//        gsl_nm2 lalgo(100, 1e-20);
        
//        bopt_params parms = initialize_parameters_to_default();
//        parms.kernel.name = "kSum(kMaternISO3,kRQISO)";
//        parms.kernel.hp_mean[0] = 1;
//        parms.kernel.hp_mean[1] = 1;
//        parms.kernel.hp_std[0] = 0.5;
//        parms.kernel.hp_std[1] = 0.5;
////        parms.kernel.hp_mean = {1, 1};
////        parms.kernel.hp_std = {0.5, 0.5};
//        parms.kernel.n_hp = 2;
//        parms.crit_name = "cHedge (cEI , cLCB , cThompsonSampling)";
//        bayesfunc bayes(parms, fdata);
//        vectord xopt(ndim);
//        bayes.optimize(xopt);
//        
//        double Ebayes = bayes.evaluateSample(xopt);
//        cout << xopt << endl;
//        cout << ::math(Ebayes) << endl;
//        exit(0);
        
        int npop = 20;
        
        energy prob0(ndim, U, J, point.mu/scale, 0);
        population pop0(prob0, npop);
        algo.evolve(pop0);
//        cout << pop0.champion().f << endl;
//        for(int q = 0; q < 10; q++) {
//        population pop0(prob0, npop);
//        algo.evolve(pop0);
//        cout << pop0.champion().f << endl;
//        }
//        cout << pop0.champion().x << endl;
//        exit(0);
        
        double E0 = DBL_MAX;
        popx = pop0.champion().x;
        fdata.theta = 0;
        try {
        localopt.optimize(popx, E0);
        } catch (std::exception& e) {
            printf("nlopt failed!: E0 refine: %d, %d\n", point.i, point.j);
            cout << e.what() << endl;
            E0 = pop0.champion().f[0];
        }
//        cout << ::math(fdata.Ei) << endl;
//        cout << ::math(E0) << endl;
//        exit(0);
        
//        lalgo.evolve(pop0);
//        cout << pop0.champion().f << endl;
//        cout << pop0.champion().x << endl;
        
//        double E0 = pop0.champion().f[0];
        
//        popx = pop0.champion().x;
        norm(popx, norms);
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                x0[2 * (i * idim + n)] = popx[2 * (i * idim + n)] / norms[i];
                x0[2 * (i * idim + n) + 1] = popx[2 * (i * idim + n) + 1] / norms[i];
            }
            transform(f0[i], f0[i] + idim, fabs[i], std::ptr_fun<const doublecomplex&, double>(abs));
            fmax[i] = *max_element(fabs[i], fabs[i] + idim);
            fn0[i] = fabs[i][1];
        }

        fres.fmin[point.i][point.j] = *min_element(fn0.begin(), fn0.end());
        fres.fn0[point.i][point.j] = fn0;
        fres.fmax[point.i][point.j] = fmax;
        fres.f0[point.i][point.j] = x0;
        results.E0res[point.i][point.j] = E0;

        energy probth(ndim, U, J, point.mu/scale, theta);
        population popth(probth, npop);
        algo.evolve(popth);
//        cout << popth.champion().f << endl;

        double Eth = DBL_MAX;
        popx = popth.champion().x;
        fdata.theta = theta;
        try {
        localopt.optimize(popx, Eth);
        } catch (std::exception& e) {
            printf("nlopt failed!: Eth refine: %d, %d\n", point.i, point.j);
            cout << e.what() << endl;
            Eth = popth.champion().f[0];
        }
//        cout << ::math(fdata.Ei) << endl;
//        cout << ::math(Eth) << endl;
        
//        lalgo.evolve(popth);
//        cout << popth.champion().f << endl;
//        cout << popth.champion().x << endl;
        
//        double Eth = popth.champion().f[0];
        
//        popx = popth.champion().x;
        norm(popx, norms);
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                x0[2 * (i * idim + n)] = popx[2 * (i * idim + n)] / norms[i];
                x0[2 * (i * idim + n) + 1] = popx[2 * (i * idim + n) + 1] / norms[i];
            }
        }

        fres.fth[point.i][point.j] = x0;
        results.Ethres[point.i][point.j] = Eth;

        energy prob2th(ndim, U, J, point.mu/scale, 2*theta);
        population pop2th(prob2th, npop);
        algo.evolve(pop2th);
//        cout << pop2th.champion().f << endl;

        double E2th = DBL_MAX;
        popx = pop2th.champion().x;
        fdata.theta = 2*theta;
        try {
        localopt.optimize(popx, E2th);
        } catch (std::exception& e) {
            printf("nlopt failed!: E0 refine: %d, %d\n", point.i, point.j);
            cout << e.what() << endl;
            E2th = pop2th.champion().f[0];
        }
//        cout << ::math(fdata.Ei) << endl;
//        cout << ::math(E2th) << endl;
        
//        lalgo.evolve(pop2th);
//        cout << pop2th.champion().f << endl;
//        cout << pop2th.champion().x << endl;
        
//        double E2th = pop2th.champion().f[0];
        
//        popx = pop2th.champion().x;
        norm(popx, norms);
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                x0[2 * (i * idim + n)] = popx[2 * (i * idim + n)] / norms[i];
                x0[2 * (i * idim + n) + 1] = popx[2 * (i * idim + n) + 1] / norms[i];
            }
        }

        fres.f2th[point.i][point.j] = x0;
        results.E2thres[point.i][point.j] = E2th;

        results.fs[point.i][point.j] = (E2th - 2 * Eth + E0) / (L * theta * theta);


        {
            boost::mutex::scoped_lock lock(progress_mutex);
            ++progress;
        }

        continue;
//        exit(0);


        //
        ////    Efuncth(ndim, &x[0], NULL, &parms);
        ////    return 0;
        //
        //        cout << "Setting up optimizer" << endl;
        nlopt_srand_time();
//        nlopt::opt opt(nlopt::GD_MLSL_LDS, ndim);
                        nlopt::opt opt(nlopt::GN_CRS2_LM, ndim);
        //                nlopt::opt opt(nlopt::GN_CRS2_LM, ndim);
        //        nlopt::opt opt(nlopt::LD_MMA, ndim);
        //                nlopt::opt opt(nlopt::LN_COBYLA, ndim);
        //        nlopt::opt opt(nlopt::LD_LBFGS, ndim);
        opt.set_lower_bounds(-1);
        opt.set_upper_bounds(1);
//        opt.set_maxtime(10);
        //        opt.set_ftol_rel(1e-5);
        //        opt.set_xtol_rel(1e-5);
        //        opt.set_population(1000000);
        //        opt.set_population(ndim+1);
        //        opt.set_population(10);
        //        opt.set_vector_storage(200);
        //        opt.set_maxtime(10);
        //                opt.set_xtol_rel(1e-14);
        //                opt.set_xtol_abs(1e-14);
        //                opt.set_ftol_rel(1e-14);
        //                opt.set_ftol_abs(1e-14);
        //        opt.set_xtol_abs()

                nlopt::opt lbopt(nlopt::LN_COBYLA, ndim);
//        nlopt::opt lbopt(nlopt::LD_LBFGS, ndim);
        lbopt.set_lower_bounds(-1);
        lbopt.set_upper_bounds(1);
        lbopt.set_ftol_rel(1e-6);
        lbopt.set_xtol_rel(1e-6);
        //        lbopt.set_vector_storage(10);
        opt.set_local_optimizer(lbopt);

                nlopt::opt lopt(nlopt::LN_COBYLA, ndim);
//        nlopt::opt lopt(nlopt::LD_LBFGS, ndim);
        lopt.set_lower_bounds(-1);
        lopt.set_upper_bounds(1);
        //        lopt.set_ftol_rel(1e-2);
        //        lopt.set_xtol_rel(1e-2);
        //        opt.set_local_optimizer(lopt);

        opt.set_min_objective(Efunc, &data);
        lopt.set_min_objective(Efunc, &data);
        //            cout << "Optimizer set up. Doing optimization" << endl;

        //        int NFC = 240;
        //        int NSDC = 80;
        //        double threshold = -1000;
        //        ProblemSetup* setup = ProblemSetup_create(ndim, NFC, NSDC, threshold);
        //        setup->userData = &parms;
        //        setup->costFunction = Ecfuncp;
        //        setup->randomSeed = time(NULL);
        //        for(int i = 0; i < ndim; i++) {
        //            setup->axes[i] = PAxis_create(-1,1);
        //            setup->axes[i]->axisPrecision = 1e-7;
        //        }
        //        
        //        double minvalue = findMinimum(setup);
        //        cout << "minvalue = " << minvalue << endl;
        //        exit(0);

        int res = 0;

        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                //                                f[i][n] = 1 / sqrt(dim);
                f[i][n] = uni(rng);
                //                f[i][n] = n == 1 ? 1 : 1e-2;//0;
                //                f[i][n] = (n == 1 || n == 2) ? 1/sqrt(2) : 0;
            }
        }

        vector<double> xmin(ndim);
//        copy(x.begin(), x.end(), xmin.begin());

        data.theta = 0;
//        double E0 = 0;
        double E0min = DBL_MAX;
        //        for (int k = 0; k < NSAMP; k++) {
        //            for (int i = 0; i < L; i++) {
        //                for (int n = 0; n <= nmax; n++) {
        //                    f[i][n] = doublecomplex(uni(rng), uni(rng));
        //                }
        //            }
        //            try {
        //                res = opt.optimize(x, E0);
        //            } catch (std::exception& e) {
        //                printf("nlopt failed!: E0: %d, %d\n", point.i, point.j);
        //                cout << e.what() << endl;
        //                E0 = numeric_limits<double>::quiet_NaN();
        //                res = -10;
        //            }
        //            if (E0 < E0min) {
        //                E0min = E0;
        //                copy(x.begin(), x.end(), xmin.begin());
        //            }
        //        }
        //        cout << "E0 = " << E0 << endl;
        //        cout << "E0min = " << E0min << endl;
        //        exit(0);

        data.Emin = DBL_MAX;
        try {
            //            res = lopt.optimize(xmin, E0);
            res = opt.optimize(x, E0);
        } catch (std::exception& e) {
            printf("nlopt failed!: E0 refine: %d, %d\n", point.i, point.j);
            cout << e.what() << endl;
            E0 = numeric_limits<double>::quiet_NaN();
            res = -10;
        }
//        copy(data.xmin.begin(), data.xmin.end(), x.begin());
        cout << "E0 = " << data.Emin << endl;
        //        double E01 = data.Emin;

        try {
//            res = lopt.optimize(data.xmin, E0);
                        res = lopt.optimize(x, E0);
        } catch (std::exception& e) {
            printf("nlopt failed!: E0 refine: %d, %d\n", point.i, point.j);
            cout << e.what() << endl;
            E0 = numeric_limits<double>::quiet_NaN();
            res = -10;
        }
//        copy(data.xmin.begin(), data.xmin.end(), x.begin());
        //        cout << "E0 = " << E0 << endl;
        //        cout << "Emin = " << E01 << "\tE0 = " << E0 << endl;
        //        exit(0);

//        vector<double> xi(2 * dim);
//        vector<double> gi(2 * dim);
//        data.x0 = vector<double>(ndim);
//        copy(x.begin(), x.end(), data.x0.begin());
//        for (int i = 0; i < L; i++) {
//            copy(x.begin() + i * 2 * dim, x.begin()+(i + 1)*2 * dim, xi.begin());
//            for (int which = 0; which < 8; which++) {
//                Ecfunci(2 * dim, i, which, xi.data(), gi.data(), &data);
////                cout << "g[" << i << "][" << which << "]=" << ::math(gi) <<";"<< endl;
//
//            }
//            //        Ecfunci(2*dim, i, xi.data(), gi.data(), &data);
////            cout << "g: " << ::math(gi) << endl;
//        }
//        for (int i = 0; i < L; i++) {
//            copy(x.begin() + i * 2 * dim, x.begin()+(i + 1)*2 * dim, xi.begin());
//                    Ecfunci(2*dim, i, xi.data(), gi.data(), &data);
////                    cout << "gi[" << i << "]=" << ::math(gi) << ";" << endl;
//        }
//        vector<double> gtot(ndim);
//        Ecfunc(ndim, x.data(), gtot.data(), &data);
//        cout << "gtot=" << ::math(gtot) << ";" << endl;

        norm(x, norms);
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                x0[2 * (i * idim + n)] = x[2 * (i * idim + n)] / norms[i];
                x0[2 * (i * idim + n) + 1] = x[2 * (i * idim + n) + 1] / norms[i];
            }
            transform(f0[i], f0[i] + idim, fabs[i], std::ptr_fun<const doublecomplex&, double>(abs));
            fmax[i] = *max_element(fabs[i], fabs[i] + idim);
            fn0[i] = fabs[i][1];
        }
        for (int i = 0; i < L; i++) {
//            double Ei = Ecfunc(i, ndim, x0.data(), NULL, &data);
//            cout << "E[" << i << "] = " << ::math(Ei) << endl;
        }
//        double Etot = Ecfunc(ndim, x0.data(), NULL, &data);
//        cout << "Etot = " << ::math(Etot) << endl;
        cout << "E0 = " << ::math(E0) << endl;

        results.res0[point.i][point.j] = res;
        fres.fmin[point.i][point.j] = *min_element(fn0.begin(), fn0.end());
        fres.fn0[point.i][point.j] = fn0;
        fres.fmax[point.i][point.j] = fmax;
        fres.f0[point.i][point.j] = x0;
        results.E0res[point.i][point.j] = E0;

        //        opt.set_min_objective(Ethfunc, &parms);

        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                //                f[i][n] = 1 / sqrt(dim);
                //                //                f[i][n] = doublecomplex(1/sqrt(dim),1/sqrt(dim));
                //                f[i][n] = uni(rng);
                //                f[i][n] = n == 1 ? 1 : 1e-2;//0;
                //                f[i][n] = (n == 1 || n == 2) ? 1/sqrt(2) : 0;
            }
        }
        data.theta = theta;
//        double Eth = 0;
        double Ethmin = DBL_MAX;
        //        for (int k = 0; k < NSAMP; k++) {
        //            for (int i = 0; i < L; i++) {
        //                for (int n = 0; n <= nmax; n++) {
        //                    f[i][n] = doublecomplex(uni(rng), uni(rng));
        //                }
        //            }
        //            try {
        //                res = opt.optimize(x, Eth);
        //            } catch (std::exception& e) {
        //                printf("nlopt failed!: Eth: %d, %d\n", point.i, point.j);
        //                cout << e.what() << endl;
        //                Eth = numeric_limits<double>::quiet_NaN();
        //                res = -10;
        //            }
        //            if (Eth < Ethmin) {
        //                Ethmin = Eth;
        //                copy(x.begin(), x.end(), xmin.begin());
        //            }
        //        }
        data.Emin = DBL_MAX;
        try {
            //            res = lopt.optimize(xmin, E0);
            res = opt.optimize(x, Eth);
        } catch (std::exception& e) {
            printf("nlopt failed!: Eth refine: %d, %d\n", point.i, point.j);
            cout << e.what() << endl;
            Eth = numeric_limits<double>::quiet_NaN();
            res = -10;
        }
//        copy(xmin.begin(), xmin.end(), x.begin());
        try {
            res = lopt.optimize(xmin, Eth);
        } catch (std::exception& e) {
            printf("nlopt failed!: Eth refine: %d, %d\n", point.i, point.j);
            cout << e.what() << endl;
            Eth = numeric_limits<double>::quiet_NaN();
            res = -10;
        }
//        copy(xmin.begin(), xmin.end(), x.begin());

        //        try {
        //            res = lopt.optimize(x, Eth);
        //        } catch (std::exception& e) {
        //            printf("nlopt failed!: Eth: %d, %d\n", point.i, point.j);
        //            cout << e.what() << endl;
        //            Eth = numeric_limits<double>::quiet_NaN();
        //            res = -10;
        //        }

        //        Encfunc(ndim, x.data(), grad.data(), &parms);
        //        transform(grad.begin(), grad.end(), grad0.begin(), std::ptr_fun<double,double>(std::abs));
        //        cout << "th: " << *min_element(grad0.begin(), grad0.end()) << " - " << *max_element(grad0.begin(), grad0.end()) << endl;

        norm(x, norms);
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                x0[2 * (i * idim + n)] = x[2 * (i * idim + n)] / norms[i];
                x0[2 * (i * idim + n) + 1] = x[2 * (i * idim + n) + 1] / norms[i];
            }
        }

        results.resth[point.i][point.j] = res;
        fres.fth[point.i][point.j] = x0;
        results.Ethres[point.i][point.j] = Eth;

        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                //                f[i][n] = 1 / sqrt(dim);
                //                f[i][n] = uni(rng);
                //                f[i][n] = n == 1 ? 1 : 1e-2;//0;
                //                f[i][n] = (n == 1 || n == 2) ? 1/sqrt(2) : 0;
            }
        }
        data.theta = 2 * theta;
        double Eth2 = 0;
        double Eth2min = DBL_MAX;
        //        for (int k = 0; k < NSAMP; k++) {
        //            for (int i = 0; i < L; i++) {
        //                for (int n = 0; n <= nmax; n++) {
        //                    f[i][n] = doublecomplex(uni(rng), uni(rng));
        //                }
        //            }
        //            try {
        //                res = opt.optimize(x, Eth2);
        //            } catch (std::exception& e) {
        //                printf("nlopt failed!: Eth2: %d, %d\n", point.i, point.j);
        //                cout << e.what() << endl;
        //                Eth2 = numeric_limits<double>::quiet_NaN();
        //                res = -10;
        //            }
        //            if (Eth2 < Eth2min) {
        //                Eth2min = Eth2;
        //                copy(x.begin(), x.end(), xmin.begin());
        //            }
        //        }
        data.Emin = DBL_MAX;
        try {
            //            res = lopt.optimize(xmin, E0);
            res = opt.optimize(x, Eth2);
        } catch (std::exception& e) {
            printf("nlopt failed!: Eth2 refine: %d, %d\n", point.i, point.j);
            cout << e.what() << endl;
            Eth2 = numeric_limits<double>::quiet_NaN();
            res = -10;
        }
//        copy(xmin.begin(), xmin.end(), x.begin());
        try {
            res = lopt.optimize(xmin, Eth2);
            //            printf("Twisted energy 2: %0.10g\n", Eth2);
        } catch (std::exception& e) {
            printf("nlopt failed!: Eth2 refine: %d, %d\n", point.i, point.j);
            cout << e.what() << endl;
            Eth2 = numeric_limits<double>::quiet_NaN();
            res = -10;
        }
//        copy(xmin.begin(), xmin.end(), x.begin());

        //        try {
        //            res = lopt.optimize(x, Eth2);
        //            //            printf("Twisted energy 2: %0.10g\n", Eth2);
        //        } catch (std::exception& e) {
        //            printf("nlopt failed!: Eth2: %d, %d\n", point.i, point.j);
        //            cout << e.what() << endl;
        //            Eth2 = numeric_limits<double>::quiet_NaN();
        //            res = -10;
        //        }

        //        Encfunc(ndim, x.data(), grad.data(), &parms);
        //        transform(grad.begin(), grad.end(), grad0.begin(), std::ptr_fun<double,double>(std::abs));
        //        cout << "th2: " << *min_element(grad0.begin(), grad0.end()) << " - " << *max_element(grad0.begin(), grad0.end()) << endl;

        norm(x, norms);
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                x0[2 * (i * idim + n)] = x[2 * (i * idim + n)] / norms[i];
                x0[2 * (i * idim + n) + 1] = x[2 * (i * idim + n) + 1] / norms[i];
            }
        }

        fres.fth2[point.i][point.j] = x0;
        results.Eth2res[point.i][point.j] = Eth2;

        results.resth2[point.i][point.j] = res;
        results.fs[point.i][point.j] = (Eth2 - 2 * Eth + E0) / (L * theta * theta);
        //        cout << "fs = " << (Eth2-2*Eth+E0)/(0.01*0.01) << endl;

        //    
        //        cout << "Eth - E0 = " << Eth-E0 << endl << endl;

        {
            boost::mutex::scoped_lock lock(progress_mutex);
            ++progress;
        }
    }

}

vector<double> nu;

/*
 *
 */
int main(int argc, char** argv) {

//    double Jarr[] = {0.156361385726566, 0.158873021202621, 0.158546862649579, \
//0.150288101696321, 0.146737021383394};
//    double Uarr[] = {1.06675425238752, 0.697472058368311, 0.847555707339779, \
//0.729007497875799, 1.49906404251572};
//    vector<double> J = vector<double>(L);
//    vector<double> U = vector<double>(L);
//    for (int i = 0; i < L; i++) {
////        J[i] = 0.01*i;
////        U[i] = 1 + 0.1*i;
//        J[i] = Jarr[i];
//        U[i] = Uarr[i];
//    }
////    funcdata2 parms = {U, J, 0.5, 0.01, vector<double>(6)};
//    funcdata2 parms = {U, J, 0, 0, vector<double>(6)};
//    double xarr[] = {0.271841556780473,0.32773307478231,0.620263685951129,0.658761079292957,-5.02045509997817e-11,9.30609333219018e-11,
//   -4.64368349433475e-10,-1.3135835332481e-10,-1.443474892878e-10,-1.02489488077154e-9,0.304615256224549,0.380178438262588,
//   0.587804271614508,0.645879275892337,-1.31488907863261e-9,-1.27533107040275e-9,7.70428171523083e-10,2.9287715479026e-10,
//   -3.53953681196986e-10,-1.11511238203068e-9,0.563276000828406,0.561770868231313,0.455050941852603,0.40007784094655,
//   -3.73176979067762e-9,-3.02077050961503e-10,4.65648893380262e-10,3.52500471782538e-10,1.27758014186449e-10,2.53644790617447e-11,
//   -3.97956372343927e-10,-5.94637981846577e-10,-2.19942042687548e-10,-3.68728577410614e-10,6.06470786659714e-11,2.33792954795437e-9,
//   0.706047945088963,0.708164034130266,7.95812629871954e-10,1.06560669452581e-9,0.0131837149863491,0.015014373474611,0.705571296546575,
//   0.70835718655062,5.23171525019576e-10,-6.86400086711088e-10,-1.41013460568954e-10,5.05986683534095e-11,-2.74890579705431e-10,
//   1.33937335928692e-10};
//    vector<double> xx(2*L*dim);
//    for(int i = 0; i < 2*L*dim; i++) {
////        xx[i] = 0.01*i;
//        xx[i] = xarr[i];
//    }
//    vector<double> grad;
//    double E = energyfunc(xx, grad, &parms);
//    cout << ::math(E) << endl;
//    cout << ::math(parms.Ei) << endl;
//    return 0;

    //    cout << ::math(cos(0.1)) << endl;
    //    cout << ::math(exp(doublecomplex(0,1)*0.1).real()) << endl;
    //    cout << ::math(exp(-doublecomplex(0,1)*0.1).real()) << endl;

//            mt19937 rng2;
//            uniform_real_distribution<> uni2(-1, 1);
//        
//            rng2.seed(time(NULL));
//            nu = vector<double>(L, 0);
//            for(int i = 0; i < L; i++) {
//                nu[i] = 0;//0.5*uni2(rng2);
//            }
//        
//            rng2.seed(0);
//            vector<double> f(2*L*dim,0);
//            for(int i = 0; i <2*L*dim; i++) {
//                f[i] = uni2(rng2);
//            }
//                vector<double> g(2*L*dim,0);
//                	funcdata parms;
//                	parms.J = vector<double>(L,0.1);
//                	parms.U = vector<double>(L,1);
//                	parms.mu = 0.5;
//                    parms.theta = 0.1;
//                double E1 = Ecfunc(2*L*dim,f.data(),g.data(),&parms);
//                cout << ::math(E1) << endl;
    //            int id = 2;
    //            for(int id = 2; id < 2*L*dim; id++) {
    //            double df = 1e-7;
    //            f[id] += df;
    //                parms.theta = 0.1;
    //            double E2 = Ecfunc(2*L*dim,f.data(),g.data(),&parms);
    ////            cout << ::math(E2) << endl;
    //            cout << ::math(g[id]) << "\t";//endl;
    //            cout << ::math((E2-E1)/df) << "\t";//endl;
    //            cout << ::math((E2-E1)/df-g[id]) << endl;
    //            f[id] -= df;
    //            }
    //            
//                return 0;

    mt19937 rng;
    uniform_real_distribution<> uni(-1, 1);

    int seed = lexical_cast<int>(argv[1]);
    int nseed = lexical_cast<int>(argv[2]);

    double xmin = lexical_cast<double>(argv[3]);
    double xmax = lexical_cast<double>(argv[4]);
    int nx = lexical_cast<int>(argv[5]);

    deque<double> x(nx);
    if (nx == 1) {
        x[0] = xmin;
    } else {
        double dx = (xmax - xmin) / (nx - 1);
        for (int ix = 0; ix < nx; ix++) {
            x[ix] = xmin + ix * dx;
        }
    }

    double mumin = lexical_cast<double>(argv[6]);
    double mumax = lexical_cast<double>(argv[7]);
    int nmu = lexical_cast<int>(argv[8]);

    deque<double> mu(nmu);
    if (nmu == 1) {
        mu[0] = mumin;
    } else {
        double dmu = (mumax - mumin) / (nmu - 1);
        for (int imu = 0; imu < nmu; imu++) {
            mu[imu] = mumin + imu * dmu;
        }
    }

    double D = lexical_cast<double>(argv[9]);
    double theta = lexical_cast<double>(argv[10]);

    int numthreads = lexical_cast<int>(argv[11]);

    int resi = lexical_cast<int>(argv[12]);

    bool canonical = lexical_cast<bool>(argv[13]);

#ifdef AMAZON
    //    path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Gutzwiller Phase Diagram");
    path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Canonical Transformation Gutzwiller");
#else
    //    path resdir("/Users/Abuenameh/Dropbox/Amazon EC2/Simulation Results/Gutzwiller Phase Diagram");
    path resdir("/Users/Abuenameh/Documents/Simulation Results/Canonical Transformation Gutzwiller");
#endif
    if (!exists(resdir)) {
        cerr << "Results directory " << resdir << " does not exist!" << endl;
        exit(1);
    }
    for (int iseed = 0; iseed < nseed; iseed++, seed++) {
        ptime begin = microsec_clock::local_time();


        ostringstream oss;
        oss << "res." << resi << ".txt";
        path resfile = resdir / oss.str();
        while (exists(resfile)) {
            resi++;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }
        if (seed < 0) {
            resi = seed;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }

        Parameter xi;
        xi.fill(1);
        //        xi.assign(1);
        rng.seed(seed);
        if (seed > -1) {
            for (int j = 0; j < L; j++) {
                xi[j] = (1 + D * uni(rng));
            }
        }

        //        rng.seed(seed);
        nu = vector<double>(L, 0);
        //        for (int i = 0; i < L; i++) {
        //            nu[i] = 0.25 * uni(rng);
        //        }

        int Lres = L;
        int nmaxres = nmax;

        boost::filesystem::ofstream os(resfile);
        printMath(os, "canonical", resi, canonical);
        printMath(os, "Lres", resi, Lres);
        printMath(os, "nmaxres", resi, nmaxres);
        printMath(os, "seed", resi, seed);
        printMath(os, "theta", resi, theta);
        printMath(os, "Delta", resi, D);
        printMath(os, "xres", resi, x);
        printMath(os, "mures", resi, mu);
        printMath(os, "xires", resi, xi);
        os << flush;

        cout << "Res: " << resi << endl;

        //        multi_array<double, 2 > fcres(extents[nx][nmu]);
        multi_array<double, 2 > fsres(extents[nx][nmu]);
        //        multi_array<double, 2> dur(extents[nx][nmu]);
        //        multi_array<int, 2> iterres(extents[nx][nmu]);
        multi_array<double, 2 > fminres(extents[nx][nmu]);
        multi_array<vector<double>, 2> fn0res(extents[nx][nmu]);
        multi_array<vector<double>, 2> fmaxres(extents[nx][nmu]);
        multi_array<vector<double>, 2> f0res(extents[nx][nmu]);
        multi_array<vector<double>, 2> fthres(extents[nx][nmu]);
        multi_array<vector<double>, 2> f2thres(extents[nx][nmu]);
        multi_array<double, 2> E0res(extents[nx][nmu]);
        multi_array<double, 2> Ethres(extents[nx][nmu]);
        multi_array<double, 2> E2thres(extents[nx][nmu]);
        multi_array<double, 2> res0(extents[nx][nmu]);
        multi_array<double, 2> resth(extents[nx][nmu]);
        multi_array<double, 2> res2th(extents[nx][nmu]);

        fresults fres = {fminres, fn0res, fmaxres, f0res, fthres, f2thres, f2thres};
        results results = {E0res, Ethres, E2thres, E2thres, fsres, res0, resth, res2th, res2th};

        progress_display progress(nx * nmu);

        //        cout << "Queueing" << endl;
        queue<Point> points;
        for (int imu = 0; imu < nmu; imu++) {
            queue<Point> rowpoints;
            for (int ix = 0; ix < nx; ix++) {
                Point point;
                point.i = ix;
                point.j = imu;
                point.x = x[ix];
                point.mu = mu[imu];
                points.push(point);
            }
        }

        phase_parameters parms;
        parms.theta = theta;
        parms.canonical = canonical;

        //        for(int i = 0; i < NPHASES; i++) {
        //            vector<doublecomplex> randphase(L*dim);
        //            for(int j = 0; j < L*dim; j++) {
        //                randphase[j] = exp(doublecomplex(0,1)*uni(rng)*M_PI);
        //            }
        //            randomphases.push_back(randphase);
        //        }

        //        cout << "Dispatching" << endl;
        thread_group threads;
        //        vector<thread> threads;
        for (int i = 0; i < numthreads; i++) {
            //                        threads.emplace_back(phasepoints, std::ref(xi), theta, std::ref(points), std::ref(f0res), std::ref(E0res), std::ref(Ethres), std::ref(fsres), std::ref(progress));
            threads.create_thread(bind(&phasepoints, boost::ref(xi), parms, boost::ref(points), boost::ref(fres), boost::ref(results), /*boost::ref(E0res), boost::ref(Ethres), boost::ref(Eth2res), boost::ref(fsres),*/ boost::ref(progress)));
        }
        //        for (thread& t : threads) {
        //            t.join();
        //        }
        threads.join_all();


        //        printMath(os, "fcres", resi, fcres);
        printMath(os, "fsres", resi, fsres);
        //                printMath(os, "dur", resi, dur);
        //        printMath(os, "iters", resi, iterres);
        printMath(os, "fn0", resi, fn0res);
        printMath(os, "fmin", resi, fminres);
        printMath(os, "fmax", resi, fmaxres);
        printMath(os, "f0res", resi, f0res);
        printMath(os, "fthres", resi, fthres);
        printMath(os, "f2thres", resi, f2thres);
        printMath(os, "E0res", resi, E0res);
        printMath(os, "res0", resi, res0);
        printMath(os, "resth", resi, resth);
        printMath(os, "res2th", resi, res2th);
        printMath(os, "Ethres", resi, Ethres);
        printMath(os, "E2thres", resi, E2thres);

        ptime end = microsec_clock::local_time();
        time_period period(begin, end);
        cout << endl << period.length() << endl << endl;

        os << "runtime[" << resi << "]=\"" << period.length() << "\";" << endl;
    }

    //    time_t start = time(NULL);
    //
    //    int ndim = 2 * L * dim;

    //    vector<double> x(ndim);
    //
    //    vector<double> U(L), J(L);
    //	for (int i = 0; i < L; i++) {
    ////		U[i] = 0.1 * sqrt(i + 1);
    ////		J[i] = 0.1 * min(i + 1, mod(i + 1) + 1)
    ////			+ 0.2 * max(i + 1, mod(i + 1) + 1);
    //        U[i] = 1;
    //        J[i] = 0.2;
    //	}
    //
    //	doublecomplex * f[L];
    //	for (int i = 0; i < L; i++) {
    //		f[i] = reinterpret_cast<doublecomplex*>(&x[2 * i * dim]);
    //	}
    //
    //	for (int i = 0; i < L; i++) {
    //		for (int n = 0; n <= nmax; n++) {
    //            f[i][n] = 1/sqrt(dim);
    //		}
    //	}
    //    
    //	parameters parms;
    //	parms.J = J;
    //	parms.U = U;
    //	parms.mu = 0.5;
    //    parms.theta = 0.1;
    //
    ////    Efuncth(ndim, &x[0], NULL, &parms);
    ////    return 0;
    //
    //    nlopt::opt opt(/*nlopt::GN_ISRES*/nlopt::LN_COBYLA/*nlopt::LD_SLSQP*/, ndim);
    ////    nlopt::opt opt(nlopt::AUGLAG/*nlopt::GN_ISRES*//*nlopt::LN_COBYLA*//*nlopt::LD_SLSQP*/, ndim);
    ////    nlopt::opt local_opt(nlopt::LN_SBPLX, ndim);
    ////    opt.set_local_optimizer(local_opt);
    //    opt.set_lower_bounds(-1);
    //    opt.set_upper_bounds(1);
    //    vector<double> ctol(L, 1e-8);
    //    opt.add_equality_mconstraint(norm2s, NULL, ctol);
    //    opt.set_min_objective(Efunc, &parms);
    //    opt.set_xtol_rel(1e-8);
    //
    //	int res = 0;
    //    
    //    double E0 = 0;
    //    try {
    //        res = opt.optimize(x, E0);
    //        printf("Found minimum: %0.10g\n", E0);
    //    }
    //    catch(exception& e) {
    //        printf("nlopt failed! %d\n", res);
    //        cout << e.what() << endl;
    //    }
    //    
    //    opt.set_min_objective(Efuncth, &parms);
    //    
    //    double Eth = 0;
    //    try {
    //        res = opt.optimize(x, Eth);
    //        printf("Found minimum: %0.10g\n", Eth);
    //    }
    //    catch(exception& e) {
    //        printf("nlopt failed! %d\n", res);
    //        cout << e.what() << endl;
    //    }
    //    
    //    cout << "Eth - E0 = " << Eth-E0 << endl << endl;
    //
    //    for(int i = 0; i < 1; i++) {
    //        for(int n = 0; n <= nmax; n++) {
    //        cout << norm(f[i][n]) << endl;
    //        }
    //        cout << endl;
    //    }

    //    vector<double> norm2is;
    //	cout << norm2(x, norm2is) << endl;
    //	cout << E0 / norm2(x, norm2is) << endl;

    //    nlopt_set_xtol_rel(opt, 1e-8);
    //
    //    double x[2] = {1.234, 5.678}; /* some initial guess */
    //    double minf; /* the minimum objective value, upon return */
    //
    //    int res = 0;
    //    if ((res = nlopt_optimize(opt, x, &minf)) < 0) {
    //        printf("nlopt failed! %d\n",res);
    //    } else {
    //        printf("found minimum at f(%g,%g) = %0.10g\n", x[0], x[1], minf);
    //    }
    //
    //nlopt_destroy(opt);


    //    time_t end = time(NULL);
    //
    //    printf("Runtime: %ld", end - start);

    return 0;
}

