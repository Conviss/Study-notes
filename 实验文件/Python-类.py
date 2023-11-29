# 在Python中，类通过 class 关键字定义，类名通用习惯为首字母大写，
# Python3中类基本都会继承于object类，语法格式如下，我们创建一个Circle圆类:
class Circle(object):  # 创建Circle类，Circle为类名
    # pass  # 此处可添加属性和方法
    pi = 3.14  # 类属性

    def __init__(self, r):  # 初始化一个属性r（不要忘记self参数，他是类下面所有方法必须的参数）
       self.r = r  # 表示给我们将要创建的实例赋予属性r赋值

    def get_area(self):
        """ 圆的面积 """
        # return self.r**2 * Circle.pi # 通过实例修改pi的值对面积无影响，这个pi为类属性的值
        return self.r ** 2 * self.pi  # 通过实例修改pi的值对面积我们圆的面积就会改变

# circle1= Circle()
# circle2= Circle()

# 类的属性分为实例属性与类属性两种。
#
# 实例属性用于区分不同的实例；
# 类属性是每个实例的共有属性。
# 区别：实例属性每个实例都各自拥有，相互独立；而类属性有且只有一份，是共有的属性。

# circle1.r = 1  # r为实例属性
# circle2.R = 2
# print(circle1.r)  # 使用 实例名.属性名 可以访问我们的属性
# print(circle2.R)

# 在定义 Circle 类时，可以为 Circle 类添加一个特殊的 __init__() 方法，
# 当创建实例时，__init__() 方法被自动调用为创建的实例增加实例属性。

circle1 = Circle(1)  # 创建实例时直接给定实例属性，self不算在内
circle2 = Circle(2)
print(circle1.r)  # 实例名.属性名 访问属性
print(circle2.r)  # 我们调用实例属性的名称就统一了

print('----未修改前-----')
print('pi=\t', Circle.pi)
print('circle1.pi=\t', circle1.pi)  #  3.14
print('circle2.pi=\t', circle2.pi)  #  3.14
print('----通过类名修改后-----')
Circle.pi = 3.14159  # 通过类名修改类属性，所有实例的类属性被改变
print('pi=\t', Circle.pi)   #  3.14159
print('circle1.pi=\t', circle1.pi)   #  3.14159
print('circle2.pi=\t', circle2.pi)   #  3.14159
print('----通过circle1实例名修改后-----')
circle1.pi=3.14111   # 实际上这里是给circle1创建了一个与类属性同名的实例属性
print('pi=\t', Circle.pi)     #  3.14159
print('circle1.pi=\t', circle1.pi)  # 实例属性的访问优先级比类属性高，所以是3.14111
# 创建了一个与类属性同名的实例属性而已，实例属性访问优先级比类属性高，所以我们访问时优先访问实例属性，它将屏蔽掉对类属性的访问
print('circle2.pi=\t', circle2.pi)  #  3.14159
print('----删除circle1实例属性pi-----')
del circle1.pi
print('pi=\t', Circle.pi)
print('circle1.pi=\t', circle1.pi)
print('circle2.pi=\t', circle2.pi)
# 可见，千万不要在实例上修改类属性，它实际上并没有修改类属性，而是给实例绑定了一个实例属性

print(circle1.get_area())  # 调用方法 self不需要传入参数，不要忘记方法后的括号  输出 3.14
# 在类的内部，使用 def 关键字来定义方法，与一般函数定义不同，类方法必须第一个参数为 self,
# self 代表的是类的实例（即你还未创建类的实例），其他参数和普通函数是完全一样。

class Person(object):
    def __init__(self, name, gender, age):
        self.name = name
        self.gender = gender
        self.age = age

# Python中的super(Net, self).__init__()是指首先找到Net的父类（比如是类NNet），
# 然后把类Net的对象self转换为类NNet的对象，然后“被转换”的类NNet对象调用自己的init函数，
# 其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，这样子类就有了父类的__init__()的那些东西
# Net类继承nn.Module，super(Net, self).__init__()就是对继承自父类nn.Module的属性进行初始化
# 子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化

class Student(Person):
    def __init__(self, name, gender, age, school, score):
        super(Student,self).__init__(name,gender,age)
        # self.name = name.upper()
        # self.gender = gender.upper()
        # self.school = school
        # self.score = score

s = Student('Alice', 'female', 18, 'Middle school', 87)
print(s.gender)
print(s.name)
