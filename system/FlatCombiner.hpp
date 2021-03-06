////////////////////////////////////////////////////////////////////////
// This file is part of Grappa, a system for scaling irregular
// applications on commodity clusters. 

// Copyright (C) 2010-2014 University of Washington and Battelle
// Memorial Institute. University of Washington authorizes use of this
// Grappa software.

// Grappa is free software: you can redistribute it and/or modify it
// under the terms of the Affero General Public License as published
// by Affero, Inc., either version 1 of the License, or (at your
// option) any later version.

// Grappa is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// Affero General Public License for more details.

// You should have received a copy of the Affero General Public
// License along with this program. If not, you may obtain one from
// http://www.affero.org/oagpl.html.
////////////////////////////////////////////////////////////////////////

#pragma once

#include "Communicator.hpp"
#include "Addressing.hpp"
#include "GlobalAllocator.hpp"
#include "LocaleSharedMemory.hpp"
#include "ParallelLoop.hpp"
#include <utility>
#include <queue>

DECLARE_bool(flat_combining);
DECLARE_bool(flat_combining_local_only);

namespace Grappa {

/// Mixin for adding common global data structure functionality, such as mirrored 
/// allocation on all cores.
// template< typename Base, Core MASTER = 0 >
// class MirroredGlobal : public Base {
//   char pad[block_size - sizeof(Base)];
// public:
//   template <typename... Args>
//   explicit MirroredGlobal(Args&&... args): Base(std::forward<Args...>(args...)) {}
//   explicit MirroredGlobal(): Base() {}
// 
//   /// Allocate and call init() on all instances of Base
//   ///
//   /// @param init Lambda of the form: void init(Base*)
//   ///
//   /// @b Example:
//   /// @code
//   ///     auto c = MirroredGlobal<Counter>::create([](Counter* c){ new (c) Counter(0); });
//   /// @endcode
//   template< typename F >
//   static GlobalAddress<MirroredGlobal<Base>> create(F init) {
//     auto a = symmetric_global_alloc<MirroredGlobal<Base>>();
//     call_on_all_cores([a,init]{ init(static_cast<Base*>(a.localize())); });
//     return a;
//   }
// 
//   /// Allocate an instance on all cores and initialize with default constructor.
//   /// 
//   /// @note Requires Base class to have a default constructor.
//   static GlobalAddress<MirroredGlobal<Base>> create() {
//     auto a = symmetric_global_alloc<MirroredGlobal>();
//     call_on_all_cores([a]{ new (a.localize()) MirroredGlobal<Base>(); });
//     return a;
//   }
// 
//   void destroy() {
//     auto a = this->self;
//     call_on_all_cores([a]{ a->~MirroredGlobal<Base>(); });
//     global_free(a);
//   }
// 
// };

enum class FCStatus { SATISFIED, BLOCKED, MATCHED };

template <typename T>
class FlatCombiner {
    
  struct Flusher {
    Flusher * next;
    T * id;
    Worker * sender;
    
    ConditionVariable cv;
    
    Flusher(T * id): id(id) { clear(); }
    ~Flusher() { id->~T(); locale_free(id); }
    void clear() {
      next = nullptr;
      id->clear();
      sender = nullptr;
      cv.waiters_ = 0;
    }
  };
  
  void free_flusher(Flusher * s) {
    if (freelist == nullptr) {
      freelist = s;
    } else {
      s->next = freelist;
      freelist = s;
    }
  }
  Flusher * get_flusher(Flusher * s) {
    Flusher * r;
    if (freelist != nullptr) {
      r = freelist;
      freelist = r->next;
      r->clear();
    } else {
      r = new Flusher(s->id->clone_fresh());
    }
    return r;
  }
  
  Flusher * current;
  size_t inflight;
  Flusher * freelist;
  
public:
  
  FlatCombiner(T * initial): current(new Flusher(initial)), inflight(0), freelist(nullptr) {}
  ~FlatCombiner() { delete current; }
  
  // template <typename... Args>
  // explicit FlatCombiner(Args&&... args)
  //   : current(new Flusher(new T(std::forward<Args...>(args...))))
  //   , inflight(nullptr)
  //   , sender(nullptr)
  // { }
  // 
  // // in case it's a no-arg constructor
  // explicit FlatCombiner(): FlatCombiner() {}

  // for arbitrary return type defined by base class
  // auto operator()(Args&&... args) -> decltype(this->T::operator()(std::forward<Args...>(args...))) {
  
  // TODO: make this do the stuff that `combine` currently does
  // should be able to do the checks for if it should send, and if it is full, swap in a new one before passing the pointer out
  inline T* operator->() const { return current->id; }
  
  template< typename F >
  void combine(F func) {
    auto s = current;
    
    FCStatus status = func(*s->id);
    if (status == FCStatus::SATISFIED) return;
    else if (status == FCStatus::MATCHED) {
      // if (s->cv.waiters_ == 0) {
      //   CHECK(s != current);
      //   s->sender = nullptr;
      // }
      // CHECK(s->cv.waiters_ != 0);
      // CHECK(impl::get_waiters(&s->cv) != s->sender);
      Grappa::signal(&s->cv);
      return;
    }
    
    if (s->id->is_full()) {
      current = get_flusher(s);
      if (s->sender == nullptr) {
        inflight++;
        DVLOG(3) << "inflight++ -> " << inflight;
        DVLOG(3) << "flushing on full";
        flush(s);
        return;
      } // otherwise someone else assigned to send...
    } else if (inflight == 0) {
      inflight++;
      DVLOG(3) << "inflight++ -> " << inflight;
      // always need at least one in flight
      if (s->sender == nullptr) {
        current = get_flusher(s);
        DVLOG(3) << "flush because none in flight";
        flush(s);
        return;
      }
    }
    
    // not my turn yet
    Grappa::wait(&s->cv);
    
    // on wake...
    if (Grappa::current_worker() == s->sender) { // I was assigned to send
      if (s == current) {
        current = get_flusher(s);
      }
      DVLOG(3) << "flush by waken worker";
      flush(s);
      return;
    }
    
  }

  void flush(Flusher * s) {
    s->sender = Grappa::current_worker(); // (if not set already)
    DVLOG(4) << "flushing (" << s->sender << "), s(" << s << ") (this:" << this << ")";
    
    s->id->sync();
    DVLOG(4) << "flushing (" << s->sender << "), s(" << s << ") (this:" << this << ")";
    
    broadcast(&s->cv); // wake our people
      
    DVLOG(4) << "flushing (" << s->sender << "), s(" << s << ") (this:" << this << ")";
    if (current->cv.waiters_ != 0 && current->sender == nullptr) {
      // atomically claim it so no one else tries to send in the meantime
      // wake someone and tell them to send
      auto n = current;
      current = get_flusher(n);
      n->sender = impl::get_waiters(&n->cv);
      signal(&n->cv);
      DVLOG(3) << "signaled " << current->sender;
    } else {
      inflight--;
    }
    s->sender = nullptr;
    free_flusher(s);
  }
};

// template< T >
// class ProxiedGlobal : public T {
//   using Master = typename T::Master;
//   using Proxy = typename T::Proxy;
// public:  
//   GlobalAddress<ProxiedGlobal<T>> self;
//   
//   template< typename... Args >
//   static GlobalAddress<ProxiedGlobal<T>> create(Args&&... args) {
//     static_assert(sizeof(ProxiedGlobal<T>) % block_size == 0,
//                   "must pad global proxy to multiple of block_size");
//     // allocate enough space that we are guaranteed to get one on each core at same location
//     auto qac = global_alloc<char>(cores()*(sizeof(ProxiedGlobal<T>)+block_size));
//     while (qac.core() != MASTER_CORE) qac++;
//     auto qa = static_cast<GlobalAddress<ProxiedGlobal<T>>>(qac);
//     CHECK_EQ(qa, qa.block_min());
//     CHECK_EQ(qa.core(), MASTER_CORE);
//     
//     // intialize all cores' local versions
//     call_on_all_cores([=]{
//       new (s.self.localize()) ProxiedGlobal<T>(args...);
//     });
// 
//     return qa;
//   }
//   
//   void destroy() {
//     auto q = shared.self;
//     global_free(q->shared.base);
//     call_on_all_cores([q]{ q->~GlobalVector(); });
//     global_free(q);
//   }
//   
// };
// 

} // namespace Grappa

