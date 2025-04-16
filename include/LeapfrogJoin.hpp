 #ifndef _LEAPFROG_H_
 #define _LEAPFROG_H_
 
 #include <algorithm>
 #include <vector>
 
 template<class RandomAccessIterator, class T>
 inline RandomAccessIterator leapfrogSeek( RandomAccessIterator first, RandomAccessIterator last, const T& val ) {
     RandomAccessIterator it;
     int bound, halfBound;
 
     bound = 1;
     it = first + bound;
     while( it < last && it[0] < val ) {
         bound *= 2;
         it = first + bound;
     }
     halfBound = bound / 2;
 
     if( (it + bound) < last )
         last = it + bound;
 
     it = std::upper_bound( first + halfBound, last, val );
  
     if(it == first)
         return it;
 
     it--;
     if( *it == val )
         return it;
     else
         return ++it;
 }
 
 template<template<class...> class C,
          class... A,
          class T = typename C<A...>::value_type,
          class RandomAccessIterator = typename C<A...>::iterator>
 void leapfrogJoin( std::vector<C<A...>>& indexes, std::vector<T>& resultSet ) {
     RandomAccessIterator its[indexes.size()];
     T value, max;
     int it;
 
     for( auto& index : indexes )
         if( index.size() == 0 )
             return;
 
     std::sort( indexes.begin(), indexes.end(),
                [] ( const C<A...>& a, const C<A...>& b ) { return *a.begin() < *b.begin(); }
     );
 
     for ( int i = 0; i < indexes.size(); i++ )
         its[i] = indexes[i].begin();
 
     max = *( its[indexes.size() - 1] );
     it = 0;
 
     while( true ) {
         value = *( its[it] );
         if( value == max ) {
             resultSet.push_back( value );
             its[it]++;
         } else {
             its[it] = leapfrogSeek( its[it], indexes[it].end(), max );
         }
         if( its[it] == indexes[it].end() )
             break;
         max = *( its[it] );
         it = ++it % indexes.size();
     }
 }
 
 template<template<class...> class C,
          class... A,
          class T = typename C<A...>::value_type>
 std::vector<T> leapfrogJoin( std::vector<C<A...>>& indexes ) {
     std::vector<T> resultSet;
 
     leapfrogJoin( indexes, resultSet );
     return resultSet;
 }
 
 #endif
 