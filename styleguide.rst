Implementation Guide
====================

Leading underscores
-------------------

Variables
~~~~~~~~~

Variables with leading underscores signify that they are internal to a class implementation and
not intended to be part of the public API. The exception is an outside class that has knowledge
of the implementation of the class of interest. In that case, the outside class may still use the
leading-underscore variable. However, this practice is generally discouraged.

Methods
~~~~~~~

Methods beginning with an underscore signify the following possibilities:

1. They are routines not intended to be part of the public API.
2. They have not been fully tested yet

Unless you know what you're doing, you should not call these kinds of methods directly.


No leading underscores
----------------------

Variables
~~~~~~~~~

Variables without the leading underscore signify the following:

1. If we are setting the variable, it is likely implemented as a property that does the proper checks (bounds checking, 
type checking, etc.)

2. If we are accessing the variable, we are probably using the public API of an object that isn't implemented in the 
current class definition. This is a common pattern for classes that use composition

3. If the usage is not one of the points above, it could be that we were lazy and haven't implemented the proper checks
on the variable yet

Methods
~~~~~~~
"Regular" methods without leading underscores are intended to be part of the public API. Use them freely!

Class methods
-------------
The return value of class methods is often ``self``. This is to allow chaining e.g. ```testobj.do_something().do_something_else()```