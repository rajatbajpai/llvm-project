! RUN: %python %S/test_errors.py %s %flang_fc1
! Extended derived types

module m1
  type :: t1
    integer :: x
    !ERROR: Component 'x' is already declared in this derived type
    real :: x
  end type
end

module m2
  type :: t1
    integer :: i
  end type
  type, extends(t1) :: t2
    !ERROR: Component 'i' is already declared in a parent of this derived type
    integer :: i
  end type
end

module m3
  type :: t1
  end type
  type, extends(t1) :: t2
    integer :: i
    !ERROR: 't1' is a parent type of this type and so cannot be a component
    real :: t1
  end type
  type :: t3
  end type
  type, extends(t3) :: t4
  end type
  type, extends(t4) :: t5
    !ERROR: 't3' is a parent type of this type and so cannot be a component
    real :: t3
  end type
end

module m4
  type :: t1
    integer :: t1
  end type
  !ERROR: Type cannot be extended as it has a component named 't1'
  type, extends(t1) :: t2
  end type
end
module m4a
  use m4
  type, extends(t1) :: t3 ! ok, inaccessible component
  end type
end

module m5
  type :: t1
    integer :: t2
  end type
  type, extends(t1) :: t2
  end type
  !ERROR: Type cannot be extended as it has a component named 't2'
  type, extends(t2) :: t3
  end type
end

module m6
  ! t1 can be extended if it is known as anything but t3
  type :: t1
    integer :: t3
  end type
  type, extends(t1) :: t2
  end type
end
subroutine s6
  use :: m6, only: t3 => t1
  !ERROR: Type cannot be extended as it has a component named 't3'
  type, extends(t3) :: t4
  end type
end
subroutine r6
  use :: m6, only: t5 => t1
  type, extends(t5) :: t6
  end type
end

module m7
  type, private :: t1
    integer :: i1
  end type
  type, extends(t1) :: t2
    integer :: i2
    integer, private :: i3
  end type
  type :: t3
    private
    integer :: i4 = 0
    procedure(real), pointer, nopass :: pp1 => null()
  end type
  type, extends(t3) :: t4
    private
    integer :: i5
    procedure(real), pointer, nopass :: pp2
  end type
end
subroutine s7
  use m7
  type(t2) :: x
  type(t4) :: y
  integer :: j
  j = x%i2
  !ERROR: PRIVATE name 'i3' is accessible only within module 'm7'
  j = x%i3
  !ERROR: PRIVATE name 't1' is accessible only within module 'm7'
  j = x%t1%i1
  !ok, parent component is not affected by PRIVATE in t4
  y%t3 = t3()
  !ERROR: PRIVATE name 'i4' is accessible only within module 'm7'
  y%i4 = 0
  !ERROR: PRIVATE name 'pp1' is accessible only within module 'm7'
  y%pp1 => null()
  !ERROR: PRIVATE name 'i5' is accessible only within module 'm7'
  y%i5 = 0
  !ERROR: PRIVATE name 'pp2' is accessible only within module 'm7'
  y%pp2 => null()
end

! 7.5.4.8(2)
module m8
  type  :: t
    integer :: i1
    integer, private :: i2
  end type
  type(t) :: y
  integer :: a(1)
contains
  subroutine s0
    type(t) :: x
    x = t(i1=2, i2=5)  !OK
  end
  subroutine s1
    a = [y%i2]  !OK
  end subroutine
end
subroutine s8
  use m8
  type(t) :: x
  !ERROR: PRIVATE name 'i2' is accessible only within module 'm8'
  x = t(2, 5)
  !ERROR: PRIVATE name 'i2' is accessible only within module 'm8'
  x = t(i1=2, i2=5)
  !ERROR: PRIVATE name 'i2' is accessible only within module 'm8'
  a = [y%i2]
end

! 7.5.4.8(2)
module m9
  interface
    module subroutine s()
    end subroutine
  end interface
  type  :: t
    integer :: i1
    integer, private :: i2
  end type
end
submodule(m9) sm8
contains
  module subroutine s
    type(t) :: x
    x = t(i1=2, i2=5)  !OK
  end
end

module m10
  type t
    integer n
   contains
    procedure :: f
    generic, private :: operator(+) => f
  end type
 contains
  type(t) function f(x,y)
    class(t), intent(in) :: x, y
    f = t(x%n + y%n)
  end function
end module
subroutine s10
  use m10
  type(t) x
  x = t(1)
  !ERROR: PRIVATE name 'operator(+)' is accessible only within module 'm10'
  x = x + x
end subroutine
