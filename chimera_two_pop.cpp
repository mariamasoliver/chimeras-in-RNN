// *********************************************************
// *** Simulations for Kuramoto Oscillators ****************
// *** Two pop set up. Look at Panaggio et. al 2016 ********
// *** Using Euler's Method   ******************************
// ***  g++ -O3 -o example chimera_in_small_pop.cpp mt.cpp *
// *********************************************************

#include <iostream>
#include <fstream>
#include <cmath>
#include <sstream>        // std::ostringstream
#include <iomanip>        //setw, setfill
#include <stdlib.h>     // strtod
#include "stdio.h"
#include "mt.h"
#include "string.h"
#include "helfer.h"      //needed for matrix&vector class
#include <time.h>
#include <vector>
using namespace std;

//***********************************************************
template <typename T>  string nameconv(T param)
{
    ostringstream convert;   // stream used for the conversion
    convert << param;      // insert the textual representation of'Number' in the characters in the stream
    string paramstr = convert.str(); // set 'paramstr' to the contents of the stream
    convert.str("");
    return paramstr;
}
//***********************************************************
// Polar-Method should be slightly faster:
// It generates a pair of normal random variables
// This has the additional advantage that you use both random numbers
template <class T>
void noise_polar (T& mt, double *x1, double *x2)
{
   double u, v, q, p;

   do
   {
      u = 2.0 * mt.random() - 1;
      v = 2.0 * mt.random() - 1;
      q  = u * u + v * v;
   }
   
   while (q == 0.0 || q >= 1.0);

   p = sqrt(-2 * log(q) / q);
   *x1 = u * p;
   *x2 = v * p;
}

// ********************************************************
// Coupling term: all to all coupled
double all_to_all_coupling(int J, VecDoub& theta, int j, double K)
{
    double coup = 0;
    for (int z = 0; z < J; z++)
    {
        coup += sin(theta[z]-(theta[j]));
    }
    return (K/J)*coup;
}

// ********************************************************
// Coupling term: non-local coupling like Laing et. al 2009
double non_local_coupling(int J, VecDoub& theta, int j, double beta, double K)
{
    double coup = 0;
    for (int z = 0; z < J; z++)
    {
        coup += (1.+K*cos(2*M_PI*abs(j-z)/J))*cos(theta[j]-theta[z] - beta);
    }
    return coup/J;
}

/* Coupling term: non-local coupling like Zakharova et. al*/
double non_local_coupling_zak (int J, int P, VecDoub theta, double act, int j)
{
    double coup = 0.;
    
     for (int z = j - P; z <= j + P; z++)
        coup += sin(theta[z]-(theta[j]+act));
     return coup;
    
}

// ********************************************************
// Coupling terms: between nodes and between populations
double pop_coupling_theta(int J, VecDoub& theta, VecDoub& phi, int j, double beta, double A)
{
    double coup = 0;
    for (int z = 0; z < J; z++)
    {
        coup += ((1.+ A)/(2*J))*cos(theta[j]-theta[z] - beta) + ((1.- A)/(2*J))*cos(theta[j]-phi[z] - beta);
    }
    return coup;
}

double pop_coupling_phi(int J, VecDoub& theta, VecDoub& phi, int j, double beta, double A)
{
    double coup = 0;
    for (int z = 0; z < J; z++)
    {
        coup += ((1.+ A)/(2*J))*cos(phi[j]-phi[z] - beta) + ((1.- A)/(2*J))*cos(phi[j]-theta[z] - beta);
    }
    return coup;
}

void KRM(VecDoub om, int j, double A, int J, VecDoub& theta, VecDoub& ftheta, VecDoub& phi, VecDoub& fphi, double beta, double TC)
{
    ftheta[j] = om[j] - pop_coupling_theta(J, theta, phi, j, beta, A);
    ftheta[j] *= TC;
    fphi[j] = om[j] - pop_coupling_phi(J, theta, phi, j, beta, A);
    fphi[j] *= TC;
}

void E(VecDoub& theta, VecDoub& ftheta, VecDoub& phi, VecDoub& fphi, int j, double dt)
{
    theta[j] += dt*ftheta[j];
    phi[j] += dt*fphi[j];
}



// ******************************************
int main (int argc, char **argv)
{
    time_t start;
    time(&start);


    /* CHOOSE BETWEEN THE TWO SYSTEMS J = 3 OR J = 25*/
    int J = 3;
    double num = 7; //if J = 3, num = 7
    
    int J = 25;
    double num = 2; //if J = 25, num = 2
   

    /* Definitions */
    double A = 0.1;
    double beta = 0.025;
    double TC = 1;
    int tfinal = 1e6;
    double dt = 0.01;
    int N = int(tfinal/dt);
    VecDoub theta(J);
    VecDoub ftheta(J);
    VecDoub phi(J);
    VecDoub fphi(J);
    VecDoub rotations_theta(J), rotations_phi(J);
    double t, c, c_2, p, p_2;
    VecDoub marcador_theta(J);
    VecDoub marcador_phi(J);
    VecDoub theta_old(J);
    VecDoub phi_old(J);
    VecDoub theta0(J);
    VecDoub phi0(J);
    double omega = 1;
    VecDoub om(J, omega);
    
    
    /* Theta and Phi intial conditions */
    string nestring = nameconv(num)+string("_initial_conditions.txt");
    char *nechar= new char[nestring.length()+1];
    strcpy(nechar,nestring.c_str());
    ifstream fthe;
    fthe.open (nechar, ios::in);
    
    int doubleJ = 2*J;
    VecDoub chimera(doubleJ);
    for (int col = 0; col < doubleJ; col++)
        fthe >> chimera[col];

    for (int col = 0; col < J; col++)
        theta0[col] = chimera[col];
    
    for (int col = J; col < doubleJ; col++)
        phi0[col-J] = chimera[col];
    

    theta = theta0;
    phi = phi0;
    
    string nostring = string("theta_A_")+nameconv(A)+string("_N_")+nameconv(J)+("_beta_")+nameconv(beta)+string("_ic_")+nameconv(num)+string("_TC_")+nameconv(TC)+string(".txt");
    char *nochar= new char[nostring.length()+1];
    strcpy(nochar,nostring.c_str());
    ofstream fitxer1;
    fitxer1.open (nochar, ios::trunc);
    
    string cstring = string("phi_A_")+nameconv(A)+string("_N_")+nameconv(J)+("_beta_")+nameconv(beta)+string("_ic_")+nameconv(num)+string("_TC_")+nameconv(TC)+string(".txt");
    char *cchar= new char[cstring.length()+1];
    strcpy(cchar,cstring.c_str());
    ofstream fitxer2;
    fitxer2.open (cchar, ios::trunc);
    
    string cotstring = string("rotations_theta_A_")+nameconv(A)+string("_N_")+nameconv(J)+("_beta_")+nameconv(beta)+string("_ic_")+nameconv(num)+string("_TC_")+nameconv(TC)+string(".txt");
    char *cotchar= new char[cotstring.length()+1];
    strcpy(cotchar,cotstring.c_str());
    ofstream fitxer3;
    fitxer3.open (cotchar, ios::trunc);
    
    string costring = string("rotations_phi_A_")+nameconv(A)+string("_N_")+nameconv(J)+("_beta_")+nameconv(beta)+string("_ic_")+nameconv(num)+string("_TC_")+nameconv(TC)+string(".txt");
    char *cochar= new char[costring.length()+1];
    strcpy(cochar,costring.c_str());
    ofstream fitxer4;
    fitxer4.open (cochar, ios::trunc);

    
    for (int i = 0; i < N; i++)
    {
        t = i*dt;
        if (i%10 == 0)
        {
            fitxer1 << t << "\t";
            fitxer2 << t << "\t";
        }
        if (i%100 == 0)
        {
            fitxer3 << t << "\t";
            fitxer4 << t << "\t";
        }
        for (int j = 0; j < J; j++)
        {
            theta_old[j] = theta[j];
            phi_old[j] = phi[j];
            KRM(om, j, A, J, theta, ftheta, phi, fphi, beta, TC);
            E(theta, ftheta, phi, fphi, j, dt);
            if (abs(theta[j]) > M_PI)
            {
                if (theta[j] > 0)
                    theta[j] -= 2*M_PI;

                else if (theta[j] < 0)
                    theta[j] += 2*M_PI;
            }
            
            if (abs(phi[j]) > M_PI)
            {
                if (phi[j] > 0)
                    phi[j] -= 2*M_PI;

                else if (phi[j] < 0)
                    phi[j] += 2*M_PI;
            }
            
            if (t == 200)
            {
                marcador_theta[j] = theta[j];
                marcador_phi[j] = phi[j];
            }
            
            if (t >= 200 && theta_old[j] < marcador_theta[j] && theta[j] > marcador_theta[j])
                rotations_theta[j] +=1;
            
            if (t >= 200 && phi_old[j] < marcador_phi[j] && phi[j] > marcador_phi[j])
                rotations_phi[j] +=1;
            
            if (i%10 == 0)
            {
                fitxer1 << theta[j] << "\t";
                fitxer2 << phi[j] << "\t";
            }
            
            if (i%100 == 0)
            {
                fitxer3 << rotations_theta[j] << "\t";
                fitxer4 << rotations_phi[j] << "\t";
            }
            
        }
        
        if (i%10 == 0)
        {
            fitxer1 << endl;
            fitxer2 << endl;
        }
        
        if (i%100 == 0)
        {
            fitxer3 << endl;
            fitxer4 << endl;
        }
    }

    time_t fin;
    time(&fin);
    cout << fin-start << endl;

    
}
