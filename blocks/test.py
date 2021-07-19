def func1(**kwargs):

   for val in kwargs.values():

      print(val)

def func2(*args):

   for i in args:

      print(i)

def func3(info,**kwargs):

   for val in kwargs.values():
      print(1)
      print(val)

   for val in info.values():

      print(val)

if __name__ == "__main__":

   dict_test = {}
   dict_test["hpara1"] = (3,4,5,"same")
   dict_test["hpara2"] = (5,9,2,"valid")

   """
   func1(**dict_test)

   func1(hpara1 =(6,7,8),hpara2=("a","b"))

   func2((7,8,9),(6,5,"same"))

   for i,val in enumerate(dict_test.values()):

      print(f"{i}:{val}")

   print(list(dict_test.keys()))

   for i in dict_test.keys():

      print(type(i))
   """
   
   func3(dict_test)

