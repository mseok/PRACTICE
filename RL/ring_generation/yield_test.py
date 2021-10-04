def f():
    x = 0
    for i in range(5):
        yield i
        x = i
        # print("shit")

x = f()
x = iter(x)
done = False
# print(next(x))
while not done:
    xx = next(x, None)
    # print(xx)
    done = xx is None

def counter(n): 
    try:
        counter = 0
        while counter <= n:
           counter += 1
           yield counter
    finally:
        print(f"Nice! You counted to {counter}")


def test_counter(): 
   for count in counter(6): 
      print(f"count={count}")
      if count > 3: 
         break

test_counter()
