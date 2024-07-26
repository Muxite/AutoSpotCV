from selenium.common.exceptions import NoSuchElementException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import requests
import threading
import time
import os
from selenium.webdriver import ActionChains
import random
# 50 images, 1 thread, 2.07s per image
# 50 images, 2 threads, 1.19s per image
# 50 images, 5 threads, 0.786s per image
# 50 images, 10 threads, 0.695s per image
os.chdir(r"D:\Github\AutoSpotCV")


def bot(term, max_runs, is_random, location, order, step):
    google_images = 'https://images.google.ca/'
    browser = webdriver.Chrome(r'chromedrivers/chromedriver'+str(order)+'.exe')
    browser.get(google_images)
    search_bar = browser.find_element(By.XPATH, '//*[@id="APjFqb"]')  # get the search bar
    search_bar.send_keys(term)  # input the word
    search_bar.send_keys(Keys.RETURN)  # press enter
    time.sleep(3)
    for j in range(max_runs):  # start at order, skip by step
        i = j*step+order
        clickable = browser.find_elements(By.XPATH, '//div[contains(@jsname, "qQjpJ")]')  # minimal time (>20ms)
        html = browser.find_element(By.TAG_NAME, "html")
        if is_random:
            chosen = random.choice(clickable)  # random image
        else:
            chosen = clickable[i]  # sequential
        if j % int(20/step) == 0:
            html.send_keys(Keys.END)
        sleep_time = 0.5
        while True:
            time.sleep(sleep_time)
            if sleep_time > 5:
                break
            try:
                action = ActionChains(browser)
                action.click(chosen)  # click to expand image
                action.perform()
                time.sleep(sleep_time)
                image = browser.find_element(By.XPATH, '//img[contains(@jsname, "kn3ccd")]')
                try:
                    image_src = image.get_attribute("src")
                    image_download = requests.get(image_src, allow_redirects=False)
                    with open(location + term + "--" + str(i) + ".png", 'wb') as file:
                        file.write(image_download.content)
                except requests.exceptions.RequestException:
                    print("download failed on" + str(i))
                break
            except NoSuchElementException:
                sleep_time += sleep_time  # add a bit more sleep time and do it again
    browser.quit()  # close driver


def menu():
    while True:
        term = str(input("TERM: "))
        loops = int(input("LOOPS: "))
        is_random = (str(input("RANDOM?[y/n] ")) == "y")
        threads_n = int(input("THREADS: "))
        location = str(input("LOCATION: "))
        if term == "0" or loops == 0 or threads_n == 0:
            break
        threads = []
        loops_per = int(loops/threads_n + 0.5)
        very_start = time.time()
        for i in range(0, threads_n):
            t = threading.Thread(target=bot, args=(term, loops_per,
                                                   is_random, location, i, threads_n))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        print(time.time() - very_start)


def alt_run(term, loops, is_random, threads_n, location):
    threads = []
    loops_per = int(loops / threads_n + 0.5)
    very_start = time.time()
    for i in range(0, threads_n):
        t = threading.Thread(target=bot, args=(term, loops_per,
                                               is_random, location, i, threads_n))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    print(time.time() - very_start)
