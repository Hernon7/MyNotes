# Python Programming Notebook

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

```
$ python static.py
Got grades: [90, 80, 85, 94, 70]
Invalid!
```

Notice how the static methods can even work together with *from_csv* calling validate using the *cls* object. Running the code above should print out an array of valid grades, and then fail on the second attempt, thus printing out "Invalid!".