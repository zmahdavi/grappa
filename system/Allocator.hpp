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

/// Generic buddy allocator. Used by GlobalAllocator to implement
/// global heap. 
///
/// If there's an error, it's probably because you are asking for more
/// memory than is available for allocation.
/// TODO: better error messages....

#ifndef __ALLOCATOR_HPP__
#define __ALLOCATOR_HPP__

#include <cstdlib>
#include <cassert>

#include <list>
#include <map>

#include <glog/logging.h>
#include <iostream>
#include <fstream>

#include <exception>

#include <boost/tuple/tuple.hpp>

typedef intptr_t AllocatorAddress;
struct AllocatorChunk;

typedef std::map< AllocatorAddress, AllocatorChunk > ChunkMap;

typedef std::list< ChunkMap::iterator > FreeList;
typedef std::map< size_t, FreeList > FreeListMap;

/// Private struct used in allocator
struct AllocatorChunk {
  bool in_use;
  AllocatorAddress address;
  size_t size;
  FreeList::iterator free_list_slot;

  AllocatorChunk( AllocatorAddress address, size_t size )
    : in_use( false )
    , address( address )
    , size( size )
    , free_list_slot( ) // no way to make this invalid by default
  { }

  std::ostream& dump( std::ostream& o ) const {
    return o << "[ chunk " << (void *) address 
             << " size " << size 
             << " in_use " << in_use 
             << " ]";
  }

};

// overload output operator for debugging
std::ostream& operator<<( std::ostream& o, const AllocatorChunk& chunk );

class Allocator;
std::ostream& operator<<( std::ostream& o, const Allocator& a );

/// Generic buddy allocator. Used by GlobalAllocator to implement
/// global heap.
class Allocator {
private:
  const AllocatorAddress base_;
  const size_t size_;

  ChunkMap chunks_; /// stores all chunks available to allocator
  FreeListMap free_lists_; /// stores 
  
  /// find the next largest power of 2
  /// from Stanford bit twiddling hacks
  int64_t next_largest_power_of_2( int64_t v ) const {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v++;
    return v;
  }

  void remove_from_free_list( const ChunkMap::iterator & it ) {
    free_lists_[ it->second.size ].erase( it->second.free_list_slot );
    if( free_lists_[ it->second.size ].size() == 0 ) {
      free_lists_.erase( it->second.size );
    }
    it->second.in_use = true;
  }

  void add_to_free_list( const ChunkMap::iterator & it ) {
    free_lists_[ it->second.size ].push_front( it );
    it->second.free_list_slot = free_lists_[ it->second.size ].begin();
    it->second.in_use = false;
  }

  ChunkMap::iterator add_to_chunk_map( const AllocatorChunk& ac ) {
    ChunkMap::iterator it;
    bool inserted = false;
    boost::tie( it, inserted ) = chunks_.insert( std::make_pair( ac.address, ac ) );
    assert( inserted );
    return it;
  }

  void try_merge_buddy_recursive( ChunkMap::iterator cmit ) {
    // compute address of buddy
    intptr_t address = cmit->second.address;
    intptr_t buddy_address = (address ^ cmit->second.size);
    DVLOG(5) << cmit->second << " buddy address " << (void *) buddy_address;

    // does it exist?
    ChunkMap::iterator buddy_iterator = chunks_.find( buddy_address );
    if( buddy_iterator != chunks_.end() && 
        buddy_iterator->second.size == cmit->second.size &&
        buddy_iterator->second.in_use == false ) {
      DVLOG(5) << "buddy found! address " << (void *) address << " buddy address " << (void *) buddy_address;

      // remove the higher-addressed chunk
      ChunkMap::iterator higher_iterator = address < buddy_address ? buddy_iterator : cmit;
      remove_from_free_list( higher_iterator );
      chunks_.erase( higher_iterator );

      // keep the the lower-addressed chunk in the map:
      // update its size and move it to the right free list
      ChunkMap::iterator lower_iterator = address < buddy_address ? cmit : buddy_iterator;
      remove_from_free_list( lower_iterator ); // should these be swapped? I think so.
      lower_iterator->second.size *= 2;
      add_to_free_list( lower_iterator );

      // see if we have more to merge
      try_merge_buddy_recursive( lower_iterator );
    }
  }
  

public:
  class Exception : public std::exception {};
  
  Allocator( void * base, int64_t size )
    : base_( reinterpret_cast< AllocatorAddress >( base ) )
    , size_( size )
    , chunks_( )
    , free_lists_( )
  { 
    // must pass a non-zero chunk to constructor
    CHECK( size != 0 ) << "Must pass a non-zero chunk size to constructor";

    DVLOG(1) << "Allocator is responsible for addresses from " << base << " to " << (void*) ((char*) base + size);

    intptr_t offset = 0;

    // add chunks
    while( size > 0 ) {
      AllocatorAddress this_chunk_offset = offset;
      size_t this_chunk_size = size;

      // is this chunk's size not a power of two?
      if( ( size & (size - 1) ) != 0 ) {
        this_chunk_size = next_largest_power_of_2( size / 2 );
        //DVLOG(5) << "Not a power of two: adding chunk at " << this_chunk_offset << " with size " << this_chunk_size;
      }
      
      ChunkMap::iterator it = add_to_chunk_map( AllocatorChunk( this_chunk_offset, this_chunk_size ) );
      add_to_free_list( it );

      size -= this_chunk_size;
      offset += this_chunk_size;
    }
  }

  /// Free a previously-allocated chunk.
  void free( void * void_address ) {
    intptr_t address = reinterpret_cast< intptr_t >( void_address ) - base_;
    ChunkMap::iterator block_to_free_iterator = chunks_.find( address );
    assert( block_to_free_iterator != chunks_.end() );
    assert( block_to_free_iterator->second.in_use == true );
    add_to_free_list( block_to_free_iterator );

    try_merge_buddy_recursive( block_to_free_iterator );
  }

  /// Allocate size bytes
  void * malloc( size_t size ) {
    int64_t allocation_size = next_largest_power_of_2( size );

    // find a chunk large enough to start splitting.
    FreeListMap::iterator flit = free_lists_.lower_bound( allocation_size );
    if( flit == free_lists_.end() ) {
      LOG(ERROR) << "Out of memory in the global heap: couldn't find a chunk of size " << allocation_size
                 << " to hold an allocation of " << size << " bytes. Can you increase --global_heap_fraction?";
      // do this with an exception rather than CHECK_NE() so the test fixture can catch it.
      throw Allocator::Exception();
    }

    int64_t chunk_size = flit->first;
    DVLOG(5) << "chunk_size is " << chunk_size;
    DVLOG(5) << "free list size is " << flit->second.size();
    ChunkMap::iterator cmit = flit->second.front(); // huh. this is broken.
    // I think there's nothing there, even though there isn't supposed to be.
    //DVLOG(5) << "cmit is " << (void*)cmit;
    if( cmit == chunks_.end() ) DVLOG(5) << "nothing found. dump:" << *this;

    // subdivide chunk until we have what we need
    while( chunk_size > allocation_size ) {

      // remove big chunk from free list
      remove_from_free_list( cmit );

      // begin chopping chunk in half
      chunk_size /= 2;
      cmit->second.size = chunk_size;

      // add to new free list
      add_to_free_list( cmit );

      // create new buddy chunk and add to chunk map and free list
      intptr_t new_chunk_address = cmit->second.address + cmit->second.size;
      cmit = add_to_chunk_map( AllocatorChunk( new_chunk_address, chunk_size ) );
      add_to_free_list( cmit );

    }

    // finally we have a chunk of the right size
    cmit->second.in_use = true;
    remove_from_free_list( cmit );
    return reinterpret_cast< void * >( cmit->second.address + base_ );
  }







  int64_t num_chunks() const {
    return chunks_.size();
  }

  int64_t total_bytes() const {
    int64_t total = 0;
    for( ChunkMap::const_iterator i = chunks_.begin(); i != chunks_.end(); ++i ) {
      total += i->second.size;
    }
    return total;
  }

  int64_t total_bytes_in_use() const {
    int64_t total = 0;
    for( ChunkMap::const_iterator i = chunks_.begin(); i != chunks_.end(); ++i ) {
      if( i->second.in_use ) {
        total += i->second.size;
      }
    }
    return total;
  }

  int64_t total_bytes_free() const {
    int64_t total = 0;
    for( ChunkMap::const_iterator i = chunks_.begin(); i != chunks_.end(); ++i ) {
      if( !( i->second.in_use ) ) {
        total += i->second.size;
      }
    }
    return total;
  }

  /// output human-readable state
  std::ostream & dump( std::ostream& o = std::cout ) const {
    o << "all chunks = {" << std::endl;
    for( ChunkMap::const_iterator i = chunks_.begin(); i != chunks_.end(); ++i ) {
      o << "   " << i->second << std::endl;
    }
    o << "}, free lists = {" << std::endl;
    for( FreeListMap::const_iterator i = free_lists_.begin(); i != free_lists_.end(); ++i ) {
      o << "   " << i->first << ":";
      for( FreeList::const_iterator j = i->second.begin(); j != i->second.end(); ++j ) {
        o << " " << (*j)->second;
      }
      o << std::endl;
    }
    return o << "}";
  }
};

#endif
