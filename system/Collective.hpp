// Copyright 2010-2012 University of Washington. All Rights Reserved.
// LICENSE_PLACEHOLDER
// This software was created with Government support under DE
// AC05-76RL01830 awarded by the United States Department of
// Energy. The Government has certain rights in the software.


#ifndef COLLECTIVE_HPP
#define COLLECTIVE_HPP

#include "Grappa.hpp"
#include "ForkJoin.hpp"
#include "common.hpp"

int64_t collective_max(int64_t a, int64_t b);
int64_t collective_min(int64_t a, int64_t b);
int64_t collective_add(int64_t a, int64_t b);
int64_t collective_mult(int64_t a, int64_t b);

#define COLL_MAX &collective_max
#define COLL_MIN &collective_min
#define COLL_ADD &collective_add
#define COLL_MULT &collective_mult

/// @deprecated, replace with 
/// void Grappa_allreduce(T*,size_t,T*)
int64_t Grappa_collective_reduce( int64_t (*commutative_func)(int64_t, int64_t), Node home_node, int64_t myValue, int64_t initialValue );

template< typename T >
inline T coll_add(const T& a, const T& b) {
  return a+b;
}

#define HOME_NODE 0
extern Thread * reducing_thread;
extern Node reduction_reported_in;

// This class is just for holding the reduction
// value in a type-safe manner
template < typename T >
class Reductions {
  private:
    Reductions() {}
  public:
    static T reduction_result;
    static T final_reduction_result;
};

template <typename T>
T Reductions<T>::reduction_result;

template <typename T> 
T Reductions<T>::final_reduction_result;

// wake the caller with the final reduction value set
template< typename T >
static void am_reduce_wake(T * val, size_t sz, void * payload, size_t psz) {
  Reductions<T>::final_reduction_result = *val;
  Grappa_wake(reducing_thread);
}

// wake the caller with the final reduction array value set
template< typename T >
static void am_reduce_array_wake(T * val, size_t sz, void * payload, size_t psz) {
  memcpy(Reductions<T*>::final_reduction_result, val, sz);
  Grappa_wake(reducing_thread);
}

// Grappa active message sent by every Node to HOME_NODE to perform reduction in one place
template< typename T, T (*Reducer)(const T&, const T&), T BaseVal>
static void am_reduce(T * val, size_t sz, void* payload, size_t psz) {
  CHECK(Grappa_mynode() == HOME_NODE);

  if (reduction_reported_in == 0) Reductions<T>::reduction_result = BaseVal;
  Reductions<T>::reduction_result = Reducer(Reductions<T>::reduction_result, *val);

  reduction_reported_in++;
  VLOG(5) << "reported_in = " << reduction_reported_in;
  if (reduction_reported_in == Grappa_nodes()) {
    reduction_reported_in = 0;
    for (Node n = 0; n < Grappa_nodes(); n++) {
      VLOG(5) << "waking " << n;
      T data = Reductions<T>::reduction_result;
      Grappa_call_on(n, &am_reduce_wake, &data);
    }
  }
}

// Grappa active message sent by every Node to HOME_NODE to perform reduction in one place
template< typename T, T (*Reducer)(const T&, const T&), T BaseVal>
static void am_reduce_array(T * val, size_t sz, void* payload, size_t psz) {
  CHECK(Grappa_mynode() == HOME_NODE);
  
  size_t nelem = sz / sizeof(T);

  if (reduction_reported_in == 0) {
    // allocate space for result
    Reductions<T*>::reduction_result = new T[nelem];
    for (size_t i=0; i<nelem; i++) Reductions<T*>::reduction_result[i] = BaseVal;
  }

  T * rarray = Reductions<T*>::reduction_result;
  for (size_t i=0; i<nelem; i++) {
    rarray[i] = Reducer(rarray[i], val[i]);
  }
  
  reduction_reported_in++;
  VLOG(5) << "reported_in = " << reduction_reported_in;
  if (reduction_reported_in == Grappa_nodes()) {
    reduction_reported_in = 0;
    for (Node n = 0; n < Grappa_nodes(); n++) {
      VLOG(5) << "waking " << n;
      Grappa_call_on(n, &am_reduce_array_wake, rarray, sizeof(T)*nelem);
    }
    delete [] Reductions<T*>::reduction_result;
  }
}

// am_reduce with no initial value
template< typename T, T (*Reducer)(const T&, const T&) >
static void am_reduce_noinit(T * val, size_t sz, void* payload, size_t psz) {
  CHECK(Grappa_mynode() == HOME_NODE);
  
  if (reduction_reported_in == 0) Reductions<T>::reduction_result = *val; // no base val
  else Reductions<T>::reduction_result = Reducer(Reductions<T>::reduction_result, *val);
  
  reduction_reported_in++;
  VLOG(5) << "reported_in = " << reduction_reported_in;
  if (reduction_reported_in == Grappa_nodes()) {
    reduction_reported_in = 0;
    for (Node n = 0; n < Grappa_nodes(); n++) {
      VLOG(5) << "waking " << n;
      T data = Reductions<T>::reduction_result;
      Grappa_call_on(n, &am_reduce_wake, &data);
    }
  }
}

/// Global reduction across all nodes, returning the completely reduced value to everyone.
/// Notes:
///  - this suffices as a global barrier across *all nodes*
///  - as such, only one instance of this can be running at a given time
///  - and it must be called by every node or deadlock will occur
///
/// @tparam T type of the reduced values
/// @tparam Reducer commutative and associative reduce function
/// @tparam BaseVal initial value, e.g. 0 for a sum
///
/// ALLNODES
template< typename T, T (*Reducer)(const T&, const T&), T BaseVal>
T Grappa_allreduce(T myval) {
  // TODO: do tree reduction to reduce amount of serialization at Node 0
  reducing_thread = CURRENT_THREAD;
  
  Grappa_call_on(0, &am_reduce<T,Reducer,BaseVal>, &myval);
  
  Grappa_suspend();
  
  return Reductions<T>::final_reduction_result;
}

// send one element for reduction
template< typename T, T (*Reducer)(const T&, const T&), T BaseVal>
void allreduce_one_message(T * array, size_t nelem, T * result = NULL) {
  const size_t maxn = 2048 / sizeof(T);
  CHECK( nelem <= maxn );
  // default is to overwrite original array
  if (!result) result = array;
  Reductions<T*>::final_reduction_result = result;

  // TODO: do tree reduction to reduce amount of serialization at Node 0
  reducing_thread = CURRENT_THREAD;
 
  Grappa_call_on(HOME_NODE, &am_reduce_array<T,Reducer,BaseVal>, array, sizeof(T)*nelem);
  Grappa_suspend();
}

/// Vector reduction. 
/// That is, result[i] = node0.array[i] + node1.array[i] + ... + nodeN.array[i], for all i.
/// ALLNODES
template< typename T, T (*Reducer)(const T&, const T&), T BaseVal>
void Grappa_allreduce(T * array, size_t nelem, T * result = NULL) {
  const size_t maxn = 2048 / sizeof(T);

  for (size_t i=0; i<nelem; i+=maxn) {
    size_t n = MIN(maxn, nelem-i);
    allreduce_one_message<T,Reducer,BaseVal>(array+i, n, result);
  }
}

/// Global reduction across all nodes, returning the completely reduced value to everyone.
/// This variant uses no initial value for the reduction. 
/// 
/// Notes:
///  - this suffices as a global barrier across *all nodes*
///  - as such, only one instance of this can be running at a given time
///  - and it must be called by every node or deadlock will occur
///
/// @tparam T type of the reduced values
/// @tparam Reducer commutative and associative reduce function
/// @tparam BaseVal initial value, e.g. 0 for a sum
/// 
/// ALLNODES
template< typename T, T (*Reducer)(const T&, const T&) >
T Grappa_allreduce_noinit(T myval) {
  // TODO: do tree reduction to reduce amount of serialization at Node 0
  reducing_thread = CURRENT_THREAD;
  
  Grappa_call_on(0, &am_reduce_noinit<T,Reducer>, &myval);
  
  Grappa_suspend();
  
  return Reductions<T>::final_reduction_result;
}

#endif // COLLECTIVE_HPP


