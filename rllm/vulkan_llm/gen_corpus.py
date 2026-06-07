#!/usr/bin/env python3
import os

lines = []

# ─── ENGLISH LITERATURE ───
lit = """
It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief, it was the epoch of incredulity, it was the season of Light, it was the season of Darkness, it was the spring of hope, it was the winter of despair, we had everything before us, we had nothing before us, we were all going direct to Heaven, we were all going direct the other way.

Call me Ishmael. Some years ago never mind how long precisely having little or no money in my purse, and nothing particular to interest me on shore, I thought I would sail about a little and see the watery part of the world.

It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.

Happy families are all alike; every unhappy family is unhappy in its own way.

All children grow up. All except one, Peter Pan.

The sky above the port was the color of television, tuned to a dead channel.

It was a bright cold day in April, and the clocks were striking thirteen.

The building was on fire, and it wasnt my fault.

In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move.

The man in black fled across the desert, and the gunslinger followed.

He was born with a gift of laughter and a sense that the world was mad.

The sun shone, having no alternative, on the nothing new.

It was love at first sight. The first time she saw him, she knew she would never be able to make him love her back.

We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.

Mother died today. Or maybe yesterday, I do not know.

She is a witch, they said. Burn her, burn the witch.

Do not go gentle into that good night. Rage, rage against the dying of the light.

To be, or not to be, that is the question. Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune, or to take arms against a sea of troubles, and by opposing end them.

All that glitters is not gold. Not all those who wander are lost.

The woods are lovely, dark and deep, but I have promises to keep, and miles to go before I sleep.

Two roads diverged in a wood, and I took the one less traveled by, and that has made all the difference.

"""

# ─── PHILOSOPHY ───
phi = """
He who fights with monsters should look to it that he himself does not become a monster. And if you gaze long into an abyss, the abyss also gazes into you. Friedrich Nietzsche

God is dead. God remains dead. And we have killed him. How shall we comfort ourselves, the murderers of all murderers? Friedrich Nietzsche

What does not kill me makes me stronger. Friedrich Nietzsche

The unexamined life is not worth living. Socrates

I think, therefore I am. Rene Descartes

The only true wisdom is in knowing you know nothing. Socrates

Happiness is not an ideal of reason but of imagination. Immanuel Kant

Act only according to that maxim whereby you can at the same time will that it should become a universal law. Immanuel Kant

Man is condemned to be free. Jean-Paul Sartre

Hell is other people. Jean-Paul Sartre

Existence precedes essence. Jean-Paul Sartre

One cannot step twice in the same river. Heraclitus

The greatest happiness of the greatest number is the foundation of morals and legislation. Jeremy Bentham

The limits of my language mean the limits of my world. Ludwig Wittgenstein

Whereof one cannot speak, thereof one must be silent. Ludwig Wittgenstein

We suffer more often in imagination than in reality. Seneca

Luck is what happens when preparation meets opportunity. Seneca

It is not that we have a short time to live, but that we waste a lot of it. Seneca

The impediment to action advances action. What stands in the way becomes the way. Marcus Aurelius

The happiness of your life depends upon the quality of your thoughts. Marcus Aurelius

Waste no more time arguing about what a good man should be. Be one. Marcus Aurelius

The only way to deal with an unfree world is to become so absolutely free that your very existence is an act of rebellion. Albert Camus

I do not believe in the system. I believe in the individual. Ayn Rand

The question is not what you look at, but what you see. Henry David Thoreau

If you tell the truth, you do not have to remember anything. Mark Twain

Never argue with stupid people, they will drag you down to their level and then beat you with experience. Mark Twain

The mystery of human existence lies not in just staying alive, but in finding something to live for. Fyodor Dostoevsky

Pain and suffering are always inevitable for a large intelligence and a deep heart. Fyodor Dostoevsky

There is no meaning to life except the meaning man gives his life by the unfolding of his powers. Erich Fromm

To know what you prefer instead of humbly saying Amen to what the world tells you you ought to prefer, is to have kept your soul alive. Robert Louis Stevenson

The individual has always had to struggle to keep from being overwhelmed by the tribe. If you try it, you will be lonely often, and sometimes frightened. But no price is too high to pay for the privilege of owning yourself. Friedrich Nietzsche

"""

# ─── GRAMMAR ───
gram = """
The English language has eight parts of speech: nouns, pronouns, verbs, adjectives, adverbs, prepositions, conjunctions, and interjections.

A noun is a word that names a person, place, thing, or idea. Examples include cat, London, freedom, and John.

A pronoun replaces a noun. Examples include he, she, it, they, and who.

A verb expresses action or a state of being. Action verbs include run, jump, and think. Linking verbs include is, am, are, was, and were.

An adjective modifies a noun or pronoun. For example, in the phrase the red car, red modifies the noun car.

An adverb modifies a verb, adjective, or another adverb. For example, in she ran quickly, quickly modifies the verb ran.

A preposition shows the relationship between a noun and another word. Examples include in, on, at, by, for, with, and about.

A conjunction connects words, phrases, or clauses. Coordinating conjunctions include and, but, or, nor, for, yet, and so.

An interjection expresses strong emotion, like wow, ouch, hey, and oh.

The subject of a sentence is the person, place, thing, or idea that is doing or being something. The predicate tells something about the subject.

A simple sentence contains one independent clause. Example: The dog barked.

A compound sentence contains two independent clauses joined by a conjunction. Example: The dog barked, and the cat ran away.

A complex sentence contains one independent clause and at least one dependent clause. Example: When the dog barked, the cat ran away.

A compound-complex sentence contains two or more independent clauses and at least one dependent clause.

The present tense describes actions happening now. The past tense describes actions that already happened. The future tense describes actions that will happen.

The present perfect tense connects the past to the present. Example: I have finished my work.

The past perfect tense describes an action completed before another past action. Example: I had finished my work before she arrived.

The future perfect tense describes an action that will be completed before a specified future time. Example: I will have finished my work by noon.

Active voice means the subject performs the action. Passive voice means the subject receives the action. Example of active: The cat chased the mouse. Example of passive: The mouse was chased by the cat.

A dangling modifier is a word or phrase that modifies a word not clearly stated in the sentence. Incorrect: Walking to school, the rain started. Correct: Walking to school, I felt the rain start.

A comma splice occurs when two independent clauses are joined by only a comma. Incorrect: I came home, I ate dinner. Correct: I came home, and I ate dinner.

A run-on sentence occurs when two or more independent clauses are joined without proper punctuation or conjunction.

Its is possessive. Its is a contraction of it is. Its cold outside means it is cold outside. The dog wagged its tail means the tail belongs to the dog.

Their is possessive. There indicates a place. Theyre is a contraction of they are. Their house is over there, and theyre happy.

Affect is usually a verb meaning to influence. Effect is usually a noun meaning a result. The medicine affected her mood. The effect was immediate.

Than is used for comparisons. Then is used for time. She is taller than him. First we ate, then we left.

"""

# ─── LINUX ───
lin = """
Linux is a family of open-source Unix-like operating systems based on the Linux kernel. The kernel was created by Linus Torvalds in 1991.

The Linux kernel manages hardware resources, processes, memory, and device drivers. It provides the interface between user programs and hardware.

The shell is a command-line interpreter that allows users to interact with the operating system. Common shells include Bash, Zsh, Fish, and Ksh.

Bash stands for Bourne Again Shell. It is the default shell on most Linux distributions.

The file system hierarchy in Linux starts at the root directory /. Important directories include /bin for essential binaries, /etc for configuration files, /home for user home directories, /var for variable data, /tmp for temporary files, and /usr for user programs.

The ls command lists directory contents. Common options include ls -l for long format, ls -a for hidden files, and ls -h for human-readable sizes.

The cd command changes the current directory. cd without arguments goes to the home directory. cd - goes to the previous directory.

The pwd command prints the current working directory.

The cp command copies files and directories. cp source destination copies a file. cp -r source destination copies a directory recursively.

The mv command moves or renames files and directories.

The rm command removes files. rm -r removes directories recursively. rm -f forces removal. Be careful with rm.

The mkdir command creates directories. mkdir -p creates parent directories as needed.

The rmdir command removes empty directories.

The cat command concatenates and displays files. It is often used to view small files.

The less command displays files one page at a time. Press space to go forward, b to go back, and q to quit.

The head command displays the first lines of a file. head -n 20 shows the first 20 lines.

The tail command displays the last lines of a file. tail -n 20 shows the last 20 lines. tail -f follows a file as it grows.

The grep command searches for patterns in files. grep pattern file searches for pattern in file. grep -r pattern searches recursively. grep -i ignores case. grep -v inverts the match.

The find command searches for files and directories. find . -name file finds by name. find . -type f finds only files. find . -mtime -1 finds files modified in the last day.

The chmod command changes file permissions. chmod 755 file sets rwxr-xr-x. chmod +x file adds execute permission.

The chown command changes file ownership. chown user:group file changes both user and group.

The ps command displays running processes. ps aux shows all processes with details.

The top command displays real-time system information and running processes.

The kill command sends a signal to a process. kill -9 PID forcefully terminates a process.

The man command displays the manual page for a command. man ls shows the manual for ls.

The sudo command executes a command as another user, typically root.

The apt command manages packages on Debian-based systems. apt update updates the package list. apt install package installs a package. apt remove package removes a package.

The systemctl command manages systemd services. systemctl start service starts a service. systemctl enable service enables a service to start at boot.

The ssh command connects to a remote machine securely. ssh user@host connects to host as user.

The scp command copies files over SSH. scp file user@host:path copies a file to a remote host.

The rsync command synchronizes files and directories between locations. rsync -av source destination syncs with archive mode and verbose output.

Network configuration in Linux uses tools like ip, ifconfig, netstat, and ss. The ip command is the modern replacement for ifconfig.

A pipe | connects the output of one command to the input of another. Example: ls -la | grep txt lists files and filters for txt.

Redirection uses > to send output to a file and < to read input from a file. >> appends to a file. 2> redirects error output.

The Linux kernel supports multiple file systems including ext4, XFS, Btrfs, ZFS, and NTFS.

A symbolic link is a special file that points to another file. ln -s target link creates a symbolic link.

Environment variables store configuration values. PATH specifies directories to search for commands. HOME specifies the user's home directory. Export makes a variable available to child processes.

The .bashrc file contains commands that run when a new Bash shell starts. The .bash_profile file runs for login shells.

A process has a PID process ID and a PPID parent process ID. Processes can run in the background with &.

The kernel space runs in ring 0 with full hardware access. User space runs in ring 3 with restricted access. System calls bridge the two.

The proc file system /proc provides a view into kernel data structures. It contains information about processes, hardware, and system configuration.

"""

# ─── CODING ───
code = """
A variable is a named storage location in memory that holds a value. In C++, variables must be declared with a type. Example: int x = 42.

Data types in C++ include int for integers, float for floating-point numbers, double for double-precision floats, char for characters, bool for boolean values, and string for text.

A function is a reusable block of code that performs a specific task. Functions have a return type, a name, parameters, and a body. Example: int add(int a, int b) { return a + b; }

A class is a user-defined type that encapsulates data and functions. Example: class Dog { public: void bark(); private: string name; };

Inheritance allows a class to derive from another class. Example: class Puppy : public Dog {};

Polymorphism allows objects of different types to be treated as objects of a common base type. Virtual functions enable dynamic dispatch in C++.

A pointer stores the memory address of another variable. Example: int* p = &x; dereferences to access the value at that address.

A reference is an alias for an existing variable. Example: int& ref = x; modifies x through ref.

The standard template library STL provides containers, iterators, and algorithms. Common containers include vector, list, map, set, and unordered_map.

A vector is a dynamic array that can grow and shrink. Example: vector<int> v = {1, 2, 3}; v.push_back(4);

A map is an associative container that stores key-value pairs. Example: map<string, int> ages; ages["Alice"] = 30;

An iterator provides a way to traverse a container. Example: for (auto it = v.begin(); it != v.end(); ++it) {}

Recursion is a technique where a function calls itself. Example: int factorial(int n) { return n <= 1 ? 1 : n * factorial(n - 1); }

Dynamic memory allocation uses new and delete in C++. Example: int* arr = new int[10]; delete[] arr;

A lambda is an anonymous function. Example: auto add = [](int a, int b) { return a + b; };

Exception handling uses try, catch, and throw. Example: try { throw runtime_error("error"); } catch (exception& e) { cout << e.what(); }

Python is dynamically typed and uses indentation for blocks. Example: def hello(name): print(f"Hello, {name}")

Python lists are ordered and mutable. Example: lst = [1, 2, 3]; lst.append(4)

Python dictionaries map keys to values. Example: d = {"key": "value"}; d["new"] = "data"

List comprehensions provide a concise way to create lists. Example: squares = [x*x for x in range(10)]

A decorator modifies the behavior of a function. Example: @staticmethod or @property or custom decorators.

Generators use yield to produce a sequence of values lazily. Example: def count(): n = 0; while True: yield n; n += 1

Context managers use the with statement for resource management. Example: with open("file.txt") as f: content = f.read()

The shell is a programming language. Variables are assigned without spaces. Example: NAME="World" && echo "Hello, $NAME"

Shell loops iterate over items. Example: for i in {1..5}; do echo $i; done

Conditionals in shell use if, then, elif, else, fi. Example: if [ -f file ]; then echo exists; fi

Git is a distributed version control system. git init creates a new repository. git clone copies a repository. git add stages changes. git commit saves changes. git push uploads commits. git pull downloads changes.

Make is a build automation tool. A Makefile specifies targets, dependencies, and commands. Example: all: program; program: main.cpp; g++ -o program main.cpp

A signal is a software interrupt delivered to a process. SIGINT is sent when the user presses Ctrl+C. SIGTERM requests termination. SIGKILL forcefully terminates.

Multithreading allows concurrent execution. pthreads are the POSIX threading library. Example: pthread_create creates a new thread. pthread_join waits for a thread.

The volatile keyword tells the compiler that a variable may change unexpectedly. Atomic operations prevent data races. Mutexes ensure exclusive access.

Memory management involves the stack for local variables and the heap for dynamic allocation. Stack allocation is fast but limited. Heap allocation is slower but flexible.

Big O notation describes algorithm complexity. O(1) is constant time. O(n) is linear time. O(n log n) is linearithmic time. O(n^2) is quadratic time.

A linked list consists of nodes where each node points to the next. A doubly linked list has pointers in both directions.

A binary tree has nodes with at most two children. A binary search tree maintains the invariant that left child values are less than the parent and right child values are greater.

Hashing maps data of arbitrary size to fixed-size values. A hash table uses a hash function to compute an index into an array of buckets.

TCP is a connection-oriented protocol that guarantees reliable delivery. UDP is connectionless and does not guarantee delivery.

The OSI model has seven layers: physical, data link, network, transport, session, presentation, and application.

REST is an architectural style for APIs. Resources are identified by URLs. HTTP methods include GET, POST, PUT, PATCH, and DELETE.

JSON is a lightweight data interchange format. Example: {"name": "John", "age": 30}

"""

# ─── LATIN / CLASSICAL ───
lat = """
Cogito ergo sum. I think, therefore I am.

Veni, vidi, vici. I came, I saw, I conquered.

Alea iacta est. The die is cast.

Carpe diem. Seize the day.

In vino veritas. In wine, there is truth.

Caveat emptor. Let the buyer beware.

Et tu, Brute? And you, Brutus?

Tempus fugit. Time flies.

E pluribus unum. Out of many, one.

Acta non verba. Actions, not words.

Ad astra per aspera. Through hardships to the stars.

Scientia potentia est. Knowledge is power.

Errare humanum est. To err is human.

Vita brevis, ars longa. Life is short, art is long.

"""

lines = [lit, phi, gram, lin, code, lat]

all_text = "\n\n".join(lines)

# Remove leading/trailing whitespace from each line and pack
all_text = "\n".join(line.strip() for line in all_text.split("\n"))
# Collapse multiple blank lines
while "\n\n\n" in all_text:
    all_text = all_text.replace("\n\n\n", "\n\n")

with open("corpus.trdat", "w") as f:
    f.write(all_text)

print(f"Generated corpus.trdat: {len(all_text)} chars, {len(all_text.split())} words")
