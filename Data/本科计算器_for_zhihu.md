# 基于Qt和C++的面向对象计算器软件开发（附源码）

在最开始学习C++的时候实现了这个小程序，对继承和多态进行了淋漓尽致的应用，初学者可以依靠本程序对面向对象的操作进行初步了解。

<u>***项目已开源，链接见文末，文案转载需标明出处！***</u>

## 1.需求分析

仿照Windows系统的计算器软件，应用堆栈和类的继承的原理，设计一款能够面向对象使用并且能够实现基本的运算的计算器。应用Qt设计出承载该计算器的通用界面。可以实现面向对象的操作，***例如添加一个新的定义运算符只需要通过继承的操作添加一个新的继承类并进行符号注册即可实现，不需要在具体的函数内部再改写。***

主要实现的有以下功能：加法，减法，乘法，除法，三角函数，开方，平方，求余，对数计算，倒数计算，阶乘计算，相反数计算，圆周率 <img src="https://www.zhihu.com/equation?tex=pi" alt="pi" class="ee_img tr_noresize" eeimg="1"> 和自然底数 <img src="https://www.zhihu.com/equation?tex=e" alt="e" class="ee_img tr_noresize" eeimg="1"> ， 全清空操作，退格操作，清空数字寄存器操作。

## 2.概要设计

该系统主要涉及的主要有符号注册模块，计算模块，差错检验模块，以及最后的总的界面设计。软件总的流程设计如图所示：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220104619706.png" alt="image-20220220104619706" style="zoom:50%;" />

键盘上可以操作将计算器输入的操作，按下等于号相当于执行计算的操作，在此期间会执行 ``isRight()``函数判断计算式输入是否正确， 如果输入错误则只能通过清空删除键才能执行下一步操作。计算完毕或者检查出输入格式错误后可以执行三种不同的删除按键，达到重新输入计算式进行计算的目的。

其中涉及的模块主要由以下几个部分：

- 计算（calculate）部分

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220105149160.png" alt="image-20220220105149160" style="zoom:50%;" />

进入计算部分后先进行判断输入的字符串是否正确，不正确的话throw一个error，正确的话再进行下面的判断，一共分为三种情况，如果是数字的话则入数字栈，如果是括号的话则判断是左括号还是右括号，左括号压栈（“#”），右括号执行calculate计算，如果为符号的话则判断优先级进行计算。

- 差错检验（isRight）部分

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220153622081.png" alt="image-20220220153622081" style="zoom:50%;" />

通过 ``isRight()``函数可以判断返回的字符串是否是合法的字符串，虽然通过以上的判断方式只能判断一部分出现的差错情况，如果想编程成为完全没有任何错误出现的差错检查还需要进一步完善该模块，但以上方法已经基本适用于许多常见的出错情况。

- 符号注册（factory）部分

想要理解factory部分还要先了解整个系统的堆栈部分和符号类部分。堆栈部分是由数字栈和符号栈组成的，数字栈里存的为double类型的数字值，符号栈里存的是``shared_pr``指针类型的指向符号基类的指针。 符号基类在本系统中定义为一个抽象类，不存在实际的意义，所有的符号的计算方式都需要通过继承的方式继承下来才可以使用。如图所示：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220154047440.png" alt="image-20220220154047440" style="zoom:50%;" />

operator类的继承关系如图所示：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220154102571.png" alt="image-20220220154102571" style="zoom:50%;" />

Factory的部分相当于将字符串类型的符号和具体的类类型（具体的符号类）的指针建立了一种映射关系，可以通过遍历factory的前半部分的字符串返回一个指针类型，该操作方便用户自己进行符号的添加，具体的表格映射关系如图所示：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220154125195.png" alt="image-20220220154125195" style="zoom:50%;" />

- 界面设计（Mainwindow）部分

界面的设计和排版在本系统中我主要用的为 BoxLayout 的布局方式，通过在界面QWidget上添加 QPushButton按键，QLineEdit显示文本框并定义和连接相关的槽函数即可达到计算器计算的的功能。

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220154316104.png" alt="image-20220220154316104" style="zoom:50%;" />

## 3.类的定义

本系统涉及到计算器的内部计算功能的部分一共定义了以下几个大类，分别是：计算类（Calculator）、符号类（Operator）、对象工厂类（Operatorfactory）、节点类（Node）和堆栈类（Stack）。下面依次详细介绍各个类的相关内容。

- **计算类（Calculator）**

```c++
class Calculator{ 
private:
	Stack<double> m_num; // 数字栈 
	Stack<shared_ptr<Operator>> m_opr; // 运算符栈 
	...
}
```

计算类中有两个私有的数据成员，分别为数字栈和运算符栈，数字栈中的数据位double类型，运算符栈中的类型为指向基类的shared智能指针。

```c++
	double readNum(const vector<string> &calcu,size_t &i); 
	bool isNum(const vector<string> &calcu,size_t &i) 
	void calculate(); 
	int isRight(const vector<string>& calcu); 
public:
	Calculator() { m_opr.push(make_shared<Hash>()); } 
public:
	double doIt(const vector<string>& calcu);
```

这是calculate中定义的数据成员函数，其中主要进行的计算函数为``calculate()``函数和``doIt()``函数，下面详细介绍其中各个函数的具体作用与用法。

1. 数字入栈函数 ``double readNum(const vector<string> &calcu,size_t &i)``

该函数通过扫描形参中给定的string类型的容器中的字符串来判断是不是数字，当扫描 到容器中的字符串的第一位为0-9的字符或者是字符p或e时（存在pi和自然底数e），则将字符串转化为具体的double类型作为返回值返回。

2. 判断数字函数 ``bool isNum(const vector<string> &calcu,size_t &i)``

该函数比较简单，如果容器中的第i个字符串为数字则返回true值。

3. 计算函数 ``void calculate()``

通过判断运算符栈栈顶的的目数，从而决定取出多少个数字栈中的数字进行计算，计算完毕后将相应的栈顶的空间进行释放。

4. 检查正误函数 ``int isRight(const vector<string>& calcu)``

判断输入的字符串的正确（合法）与否，如果合法则返回一个 1，非法返回 0。

5. 执行函数 ``double doIt(const vector<string>& calcu)``

该函数主要是进行入栈操作和调用``calculate``函数返回最终值的作用，通过轮回扫描字符串，发现当下的字符是算术运算符还是数字，从而执行相应的入栈操作，最后将总的运算结果以 double 的形式返回。

- **节点类（Node）**

```c++
template<typename T> 
class Node { 
	friend class Stack<T>; 
	T m_value; 
	Node* m_next = nullptr; 
public:
	Node() = default; // 默认构造函数 
	Node(const Node& rhs) = delete; // 禁用拷贝构造函数 
	Node& operator =(const Node& rhs) = delete; 
	Node(const T& val) :m_value(val) {} // 含参构造函数 
	Node(T&& val) :m_value(std::move(val)) { } // 含参移动构造函数 
	const T& value() const { return m_value; } 
	Node* next() { return m_next; }
};
```

节点类为本系统的基础，因为所有的堆栈都是有一个个的节点所组成的，每个节点对象存在两个数据成员，分别是其中存在的具体值``m_value``和相对应的节点指针，所以该类对应的函数也大部分都是构造函数或者是返回其类的数据成员的函数值得一提的是，由于本系统存在两个不同类型的栈，栈内节点的数据也是不同类型，所以调用模板进行节点类的定义。

- **堆栈类（Stack）**

```c++
template<typename T> 
class Stack { 
	Node<T>* m_top = nullptr; 
public:
	Stack() = default; 
	Stack(const Stack&) = delete; 
	Stack& operator=(const Stack&) = delete; 
	~Stack(); void clear(); 
	void push(const T& val); 
	void push(T&& val); \
	void pop(); 
	bool empty()const ； 
	const T& top() ；
};
```

堆栈类是由若干个节点互相相连而组成的一个类，其数据成员只有一个节点类的模板指针，用来指向节点相连之后的顶层的位置，从而形成类似堆栈的形状。其中的``push()``函数用来向堆栈中添加一个节点，``pop()``函数是删除最顶端的节点，``clear()``函数是将整个堆栈全部清空，``empty()``函数用来判断堆栈是否清空。

- **符号类（Operator）**

```c++
class Operator { 
public:
	Operator(string c, int numOprd, int pre) :m_symbol(c), m_numOprand(numOprd), m_precedence(pre) {} 
	string symbol() const { return m_symbol; } 
	int numOprand() const { return m_numOprand; } 
	int precedence() const { return m_precedence; } 
	virtual double get(double a, double b) const = 0; 
	virtual ~Operator() {} 
protected:
	const string m_symbol; // 运算符符号 
	const int m_numOprand; // 运算符目数 
	const int m_precedence; };
```

符号类是一个抽象类，其不存在实际的具体化的类对象，符号类具有三个数据成员，分别是string类型的符号名，int类型的符号运算目数以及int类型的符号运算优先级。

其成员函数包含三个和数据成员名称类似的函数，目的是用来返回其数据成员的值，还有一个``geta()``纯虚函数，可以通过继承该类的派生类调用该函数达到返回具体运算符号的计算结果的作用。例如加法Plus这个派生类：

```c++
class Plus : public Operator { // 运算符+ 
public:
	Plus() :Operator("+", 2, 2) {} 
	double get(double a, double b) const { 
		return a + b; 
	} 
};
```

如图所示为加法类将符号类继承下来，其运算符号为＂＋＂，运算目数为2目，优先级为 2。可以看见``get()``函数也被继承了下来有了具体的操作值，返回的便是``a+b``的值，如果是单目运算符的话则单独对b进行操作。

- **对象工厂类（factory）**

```c++
#define REGISTRAR(T, Key)	Factory::RegisterClass<T> reg_##T(Key);
class Factory {
public:
	template<typename T>
	struct RegisterClass {
		RegisterClass(string opr) {
				Factory::ms_operator.emplace(opr, [] {return make_shared<T>(); });
		} 
	};
	static shared_ptr<Operator> create(string opr); 
	static map<string, function<shared_ptr<Operator>()>> ms_operator;
};
```

对象工厂类是本代码设计的灵魂部分，也是实现面向对象功能的基本操作，解决的思路主要是根据运算符的名字来自动创建相应的运算符对象，从而实现一个类注册的机制，操作者在完成一个运算符类的定以后便可以调用 define 定义的定时来注册改符号。

其数据成员是一个map类型的静态数据成员``ms_operator``，通过定义这样一个成员可以实现string类型到智能指针的一个映射关系，这意味着只要调用和string绑定的function对象，便可以自动生成一个指向该运算符的``shared``智能指针。

其中还定义了一个嵌套类，通过该嵌套类可以进行构造函数进行构造，``emplace``用于向map中插入一组映射，达到构建一整张符号映射表的目的，其中的关键字为string类型， 值为一个``lamda``表达式用来返回式子中基类``shared``指针的右值对象其中利用``create``进行扫描时，给定值是一个string类型，通过在map表中进行遍历，找到对应的运算符后返回lamda表达式的返回值，即一个智能指针的右值对象。

通过宏定义可以方便地进行静态成员的构造，即调用构造函数，通过``REGISTRAR``创建一 个嵌套类``RegisterClass``的全局对象，同时完成``T``的注册，例如定义``Hash(“#”)``类。

``REGISTRAR(Hash,”#”)``等价于``Factory::RegisterClass<Hash>reg_Hash(“#”)``，从而创建了一个对应的 map 对象，``##``是用来连接两个语言符号，即创建的对象的名字为``reg_Hash``，修改过后的函数，版本更加简洁，而且不用修改``calculator ``里的内容就可以添加新的符号，实现了一个通用计算器的框架的搭建。

## 4.界面设计

界面设计主要在``mainwindow.h``和``mainwindow.cpp``中添加函数，其中包括定义的按键button和显示用的文本框lineedit，以及连接按键的对应的槽函数。

```c++
class MainWindow : public QMainWindow 
{
	Q_OBJECT 
public:
	MainWindow(QWidget *parent = nullptr);
	~MainWindow(); 
public slots:
	void button_0(); 
private:
	QPushButton* m_0; 
};
```

例如用``button_0``来距离，如上图所示为将按键0进行按键的定义和槽函数的声明，其中按键定义用的为 ``QPushButton``定义，显示用的文本框用的为``QLineEdit``。

```c++
MainWindow::MainWindow(QWidget *parent) 
	: QMainWindow(parent) 
{ 
	m_0=new QPushButton("0",this); 
	m_in=new QLineEdit(this); 
	...
	m_in->setReadOnly(true); 
	m_in->setAlignment(Qt::AlignRight); 
	m_in->setFrame(false); 
	m_0->setShortcut(QKeySequence(Qt::Key_0)); 
	...
	m_0->setFixedHeight(30); 
	m_in->setFixedHeight(50); 
	... 
	connect(m_0,&QPushButton::clicked,this,&MainWindow::button_0); 
	...
};
```

如上图所示为定义一个按键和一个文本框的基本过程和按键和槽函数的连接过程，其中的 ``sexFixedHight()``函数用来定义按键和文本框的默认高度。通过上面的代码可以实现按键的基本定义，其中``QPushButton``后面赋予的动态内存的``this``指针指向当下按键的分配位置，可以写也可以不写，如果写上当最后析构整张表时即可按顺序一次析构所有按键和文本框（类似于二叉树一样遍历析构），如果不加``this``的话需要在``mainwindow``的析构函数中认为加上相对应的``delete``，当按键全部定义完后即可对按键进行排版。

## 5.测试结果

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220173139827.png" alt="image-20220220173139827" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220173242490.png" alt="image-20220220173242490" style="zoom:50%;" />

按键中的所有功能均可以实现，在此便不再一一展示了。

部分差错检验：

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220173314764.png" alt="image-20220220173314764" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/QiTianyu-0403/Markdown4Zhihu/master/Data/本科计算器/image-20220220173339251.png" alt="image-20220220173339251" style="zoom:50%;" />

差错一共分为了三大类，第一种是运算符符号使用有误，输出文本框内会显示”error”。第二种是数字输入有误，文本框内会显示”error1”。第三种是运算逻辑有误，文本框内会输出”error2”，在发生错误后，输出文本框的边框会变为红色，并且会有“输入格式错误！”的字样浮现出来，此时操作者需按下 AC 键或回退键文本框方可恢复原样并可以继续键入。

## 6.结论

本项目未来可以优化的部分在于，模块设计的部分还可以更加面向对象一些，比如对于按键和槽函数连接的部分可以用一个函数代替，没有必要将多有的按键都重复定义。对于差错检验部分，还可以再规范一些。

以上就是关于本项目的全部介绍，本身实现不是特别困难，项目中也还存在一些小bug（差错检验），感兴趣的小伙伴可以继续优化。

## 7.代码code

code：