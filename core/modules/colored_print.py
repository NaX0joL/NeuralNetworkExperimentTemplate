COLORS = {
    "RED": "\033[31m",
    "GREEN": "\033[32m",
    "YELLOW": "\033[33m",
    "BLUE": "\033[34m",
    "MAGENTA": "\033[35m",
    "CYAN": "\033[36m",
}

RESET = "\033[0m"



def print_alternate_color(input:str, end="\n") -> None:
    color_name = list(COLORS.keys())
    for index, char in enumerate(input):
        print(f"{COLORS[color_name[index%len(color_name)]]}{char}{RESET}", end="")
    print(end=end)
    return


def print_colored(input:str, color:str, end="\n") -> None:
    color = color.upper()
    if color in COLORS:
        print(f"{COLORS[color]}{input}{RESET}", end=end)
    else:
        raise TypeError("print colored error! invalid color")
    return