# Python Programming Notebook

## Functions

### -divmod() in Python and its application

The `divmod()`  method in python takes two numbers and returns a pair of numbers consisting of their quotient and remainder.

```
divmod(x, y)
x and y : x is numerator and y is denominator
x and y must be non complex
```

```python
Input : x = 9, y = 3
Output :(3, 0)

Input : x = 8, y = 3
Output :(2, 2)
```

### Mathematical functions

:star:`math.ceil(x)`:Return the ceiling of *x* as a float, the smallest integer value greater than or equal to *x*.

## Decorator

### *@classmethod*

#### A typical class method might look like this:

```python
class Student(object):

    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

scott = Student('Scott',  'Robinson')
```

#### Similar *@classmethod* method would be used like this instead:

```python
class Student(object):

    @classmethod
    def from_string(cls, name_str):
        first_name, last_name = map(str, name_str.split(' '))
        student = cls(first_name, last_name)
        return student

scott = Student.from_string('Scott Robinson')
```

This follows the static factory pattern very well, encapsulating the parsing logic inside of the method itself.

The above example is a very simple one, but you can imagine more complicated examples that make this more attractive. Imagine if a Student object could be serialized in to many different formats. You could use this same strategy to parse them all:

```python
class Student(object):

    @classmethod
    def from_string(cls, name_str):
        first_name, last_name = map(str, name_str.split(' '))
        student = cls(first_name, last_name)
        return student

    @classmethod
    def from_json(cls, json_obj):
        # parse json...
        return student

    @classmethod
    def from_pickle(cls, pickle_file):
        # load pickle file...
        return student
```

The decorator becomes even more useful when you realize its usefulness in sub-classes. Since the class object is given to you within the method, you can still use the same *@classmethod* for sub-classes as well.

### *@staticmethod*

The *@staticmethod* decorator is similar to *@classmethod* in that it can be called from an uninstantiated class object, although in this case there is no *cls* parameter passed to its method. So an example might look like this:

```python
class Student(object):

    @staticmethod
    def is_full_name(name_str):
        names = name_str.split(' ')
        return len(names) > 1

Student.is_full_name('Scott Robinson')   # True
Student.is_full_name('Scott')            # False

```

Since no *self* object is passed either, that means we also don't have access to any instance data, and thus this method can not be called on an instantiated object either.

These types of methods aren't typically meant to create/instantiate objects, but they may contain some type of logic pertaining to the class itself, like a helper or utility method.

### *@classmethod* vs *@staticmethod*

The most obvious thing between these decorators is their ability to create static methods within a class. These types of methods can be called on uninstantiated class objects, much like classes using the *static* keyword in Java.

There is really only one difference between these two method decorators, but it's a major one. You probably noticed in the sections above that *@classmethod* methods have a *cls* parameter sent to their methods, while *@staticmethod* methods do not.

### A Longer Example

```python
# static.py

class ClassGrades:

    def __init__(self, grades):
        self.grades = grades

    @classmethod
    def from_csv(cls, grade_csv_str):
        grades = map(int, grade_csv_str.split(', '))
        cls.validate(grades)
        return cls(grades)


    @staticmethod
    def validate(grades):
        for g in grades:
            if g < 0 or g > 100:
                raise Exception()

try:
    # Try out some valid grades
    class_grades_valid = ClassGrades.from_csv('90, 80, 85, 94, 70')
    print 'Got grades:', class_grades_valid.grades

    # Should fail with invalid grades
    class_grades_invalid = ClassGrades.from_csv('92, -15, 99, 101, 77, 65, 100')
    print class_grades_invalid.grades
except:
    print 'Invalid!'

```

```cmd
$ python static.py
Got grades: [90, 80, 85, 94, 70]
Invalid!
```

Notice how the static methods can even work together with *from_csv* calling validate using the *cls* object. Running the code above should print out an array of valid grades, and then fail on the second attempt, thus printing out "Invalid!".

## Python’s _super()_ Function

### Overview

High level super() gives you access to methods in a superclass from the subclass that inherits from it.\
_super()_ alone returns a temporary object of the superclass that then allows you to call that superclass’s methods.\
Calling the previously built methods with super() saves you from needing to rewrite those methods in your subclass, and allows you to swap out superclasses with minimal code changes.

### _super()_ in Single Inheritance

Inheritance is a concept in object-oriented programming in which a class derives (or **inherits**) attributes and behaviors from another class without needing to implement them again.

A class exaple:

```python
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

class Square:
    def __init__(self, length):
        self.length = length

    def area(self):
        return self.length * self.length

    def perimeter(self):
        return 4 * self.length
```

You can use them as below:

```python
>>> square = Square(4)
>>> square.area()
16
>>> rectangle = Rectangle(2,4)
>>> rectangle.area()
8
```
In this example, you have two shapes that are related to each other: a square is a special kind of rectangle. The code, however, doesn’t reflect that relationship and thus has code that is essentially repeated.

By using inheritance, you can reduce the amount of code you write while simultaneously reflecting the real-world relationship between rectangles and squares:

```python
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)s
```

Here, you’ve used super() to call the __init__() of the Rectangle class, allowing you to use it in the Square class without repeating code. Below, the core functionality remains after making changes:

```python
>>> square = Square(4)
>>> square.area()
16
```

In this example, Rectangle is the superclass, and Square is the subclass.

### What Can _super()_ Do for You?

Like in other object-oriented languages, it allows you to call methods of the superclass in your subclass. The primary use case of this is to extend the functionality of the inherited method.

In the example below, you will create a class Cube that inherits from Square and extends the functionality of _.area()_ (inherited from the Rectangle class through Square) to calculate the surface area and volume of a Cube instance:

```python
class Square(Rectangle):
    def __init__(self, length):
        super().__init__(length, length)

class Cube(Square):
    def surface_area(self):
        face_area = super().area()
        return face_area * 6

    def volume(self):
        face_area = super().area()
        return face_area * self.length
```

Now that you’ve built the classes, let’s look at the surface area and volume of a cube with a side length of 3:

```python
>>> cube = Cube(3)
>>> cube.surface_area()
54
>>> cube.volume()
27
```