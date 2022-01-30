#ifndef _HELFER_H_
#define _HELFER_H_

#include <cmath>
#include <limits>
#include <cstdlib>
using namespace std;

// Hilfsfunktion: Quadrat
// **********
template<class T>
inline T SQR(const T a)
{
	return a*a;
}

// Hilfsfunktion: Maximum
// **********
template<class T>
inline const T& MAX(const T& a, const T& b)
{
	return b > a ? b : a;
}

// Hilfsfunktion: Minimum
// **********
template<class T>
inline const T& MIN(const T& a, const T& b)
{
	return b < a ? b : a;
}

// Hilfsfunktion: Vertauschen
// **********
template<class T>
inline void SWAP(T& a, T& b)
{
	T dummy = a;
	a = b;
	b = dummy;
}

// Hilfsfunktion: Signum
// **********
template<class T>
inline T SIGN(const T &a, const T &b)
{
	return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

// **********
template <class T>
class Vector
{
	private:
	int nn;
	T *v;

	public:
	Vector();
	explicit Vector(int n);
	Vector(int n, const T &a);
	Vector(int n, const T *a);
	Vector(const Vector &rhs);
	Vector & operator=(const Vector &rhs);
	typedef T value_type;
	inline T & operator[](const int i);
	inline const T & operator[](const int i) const;
	inline int size() const;
	void resize(int newn);
	void assign(int newn, const T &a);
	~Vector();
};

// **********
template <class T>
Vector<T>::Vector() : nn(0), v(NULL) {}

template <class T>
Vector<T>::Vector(int n) : nn(n), v(n>0 ? new T[n] : NULL) {}

template <class T>
Vector<T>::Vector(int n, const T& a) : nn(n), v(n>0 ? new T[n] : NULL)
{
	for(int i=0; i<n; i++) v[i] = a;
}

template <class T>
Vector<T>::Vector(int n, const T *a) : nn(n), v(n>0 ? new T[n] : NULL)
{
	for(int i=0; i<n; i++) v[i] = *a++;
}

template <class T>
Vector<T>::Vector(const Vector<T> &rhs) : nn(rhs.nn), v(nn>0 ? new T[nn] : NULL)
{
	for(int i=0; i<nn; i++) v[i] = rhs[i];
}

template <class T>
Vector<T> & Vector<T>::operator=(const Vector<T> &rhs)
{
	if (this != &rhs)
	{
		if (nn != rhs.nn) {
			if (v != NULL) delete [] (v);
			nn=rhs.nn;
			v= nn>0 ? new T[nn] : NULL;
		}
		for (int i=0; i<nn; i++)
			v[i]=rhs[i];
	}
	return *this;
}

template <class T>
inline T & Vector<T>::operator[](const int i)
{
	return v[i];
}

template <class T>
inline const T & Vector<T>::operator[](const int i) const
{
	return v[i];
}

template <class T>
inline int Vector<T>::size() const
{
	return nn;
}

template <class T>
void Vector<T>::resize(int newn)
{
	if (newn != nn) {
		if (v != NULL) delete[] (v);
		nn = newn;
		v = nn > 0 ? new T[nn] : NULL;
	}
}

template <class T>
void Vector<T>::assign(int newn, const T& a)
{
	if (newn != nn) {
		if (v != NULL) delete[] (v);
		nn = newn;
		v = nn > 0 ? new T[nn] : NULL;
	}
	for (int i=0;i<nn;i++) v[i] = a;
}

template <class T>
Vector<T>::~Vector()
{
	if (v != NULL) delete[] (v);
}

// **********
template <class T>
class Matrix
{
	private:
	int nn;
	int mm;
	T **v;

	public:
	Matrix();
	Matrix(int n, int m);
	Matrix(int n, int m, const T &a);
	Matrix(int n, int m, const T *a);
	Matrix(const Matrix &rhs);
	Matrix & operator=(const Matrix &rhs);
	typedef T value_type;
	inline T* operator[](const int i);
	inline const T* operator[](const int i) const;
	inline int nrows() const;
	inline int ncols() const;
	void resize(int newn, int newm);
	void assign(int newn, int newm, const T &a);
	~Matrix();
};

template <class T>
Matrix<T>::Matrix() : nn(0), mm(0), v(NULL) {}

template <class T>
Matrix<T>::Matrix(int n, int m) : nn(n), mm(m), v(n>0 ? new T*[n] : NULL)
{
	int i,nel=m*n;
	if (v) v[0] = nel>0 ? new T[nel] : NULL;
	for (i=1;i<n;i++) v[i] = v[i-1] + m;
}

template <class T>
Matrix<T>::Matrix(int n, int m, const T &a) : nn(n), mm(m), v(n>0 ? new T*[n] : NULL)
{
	int i,j,nel=m*n;
	if (v) v[0] = nel>0 ? new T[nel] : NULL;
	for (i=1; i< n; i++) v[i] = v[i-1] + m;
	for (i=0; i< n; i++) for (j=0; j<m; j++) v[i][j] = a;
}

template <class T>
Matrix<T>::Matrix(int n, int m, const T *a) : nn(n), mm(m), v(n>0 ? new T*[n] : NULL)
{
	int i,j,nel=m*n;
	if (v) v[0] = nel>0 ? new T[nel] : NULL;
	for (i=1; i< n; i++) v[i] = v[i-1] + m;
	for (i=0; i< n; i++) for (j=0; j<m; j++) v[i][j] = *a++;
}

template <class T>
Matrix<T>::Matrix(const Matrix &rhs) : nn(rhs.nn), mm(rhs.mm), v(nn>0 ? new T*[nn] : NULL)
{
	int i,j,nel=mm*nn;
	if (v) v[0] = nel>0 ? new T[nel] : NULL;
	for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
	for (i=0; i< nn; i++) for (j=0; j<mm; j++) v[i][j] = rhs[i][j];
}

template <class T>
Matrix<T> & Matrix<T>::operator=(const Matrix<T> &rhs)
{
	if (this != &rhs) {
		int i,j,nel;
		if (nn != rhs.nn || mm != rhs.mm) {
			if (v != NULL) {
				delete[] (v[0]);
				delete[] (v);
			}
			nn=rhs.nn;
			mm=rhs.mm;
			v = nn>0 ? new T*[nn] : NULL;
			nel = mm*nn;
			if (v) v[0] = nel>0 ? new T[nel] : NULL;
			for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
		}
		for (i=0; i< nn; i++) for (j=0; j<mm; j++) v[i][j] = rhs[i][j];
	}
	return *this;
}

template <class T>
inline T* Matrix<T>::operator[](const int i)	//subscripting: pointer to row i
{
	return v[i];
}

template <class T>
inline const T* Matrix<T>::operator[](const int i) const
{
	return v[i];
}

template <class T>
inline int Matrix<T>::nrows() const
{
	return nn;
}

template <class T>
inline int Matrix<T>::ncols() const
{
	return mm;
}

template <class T>
void Matrix<T>::resize(int newn, int newm)
{
	int i,nel;
	if (newn != nn || newm != mm) {
		if (v != NULL) {
			delete[] (v[0]);
			delete[] (v);
		}
		nn = newn;
		mm = newm;
		v = nn>0 ? new T*[nn] : NULL;
		nel = mm*nn;
		if (v) v[0] = nel>0 ? new T[nel] : NULL;
		for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
	}
}

template <class T>
void Matrix<T>::assign(int newn, int newm, const T& a)
{
	int i,j,nel;
	if (newn != nn || newm != mm) {
		if (v != NULL) {
			delete[] (v[0]);
			delete[] (v);
		}
		nn = newn;
		mm = newm;
		v = nn>0 ? new T*[nn] : NULL;
		nel = mm*nn;
		if (v) v[0] = nel>0 ? new T[nel] : NULL;
		for (i=1; i< nn; i++) v[i] = v[i-1] + mm;
	}
	for (i=0; i< nn; i++) for (j=0; j<mm; j++) v[i][j] = a;
}

template <class T>
Matrix<T>::~Matrix()
{
	if (v != NULL) {
		delete[] (v[0]);
		delete[] (v);
	}
}

// **********
typedef Vector<double> VecDoub;
typedef Vector<int> VecInt;
typedef Matrix<double> MatDoub;
typedef Matrix<int> MatInt;

#endif
