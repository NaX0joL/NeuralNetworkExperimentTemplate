import time



def use_timer(func):
    def wrapper(self, *args, **kwargs):
        start = time.time()
        output = func(self, *args, **kwargs)
        end = time.time()
        
        if kwargs["timer"]:
            print(f"time elapsed: {(end - start):.1f} seconds")
        return output
    return wrapper 