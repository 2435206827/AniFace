from colorama import Fore

color_dict = {
    "white": Fore.WHITE,
    "black": Fore.BLACK,
    "red": Fore.RED,
    "yellow": Fore.YELLOW,
    "green": Fore.GREEN,
    "cyan": Fore.CYAN,
    "blue": Fore.BLUE,
    "magenta": Fore.MAGENTA,
    "white_ex": Fore.LIGHTWHITE_EX,
    "black_ex": Fore.LIGHTBLACK_EX,
    "red_ex": Fore.LIGHTRED_EX,
    "yellow_ex": Fore.LIGHTYELLOW_EX,
    "green_ex": Fore.LIGHTGREEN_EX,
    "cyan_ex": Fore.LIGHTCYAN_EX,
    "blue_ex": Fore.LIGHTBLUE_EX,
    "magenta_ex": Fore.LIGHTMAGENTA_EX
}

def form(msg: str, endwith = "<white>"):
    msg += endwith
    for color_txt in color_dict.keys():
        if msg.find("<" + color_txt + ">") != -1:
            msg = msg.replace("<" + color_txt + ">", color_dict[color_txt])
    return msg