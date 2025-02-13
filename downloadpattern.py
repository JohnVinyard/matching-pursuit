from selenium import webdriver
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement
import numpy as np
import os
import librosa
from matplotlib import pyplot as plt
from time import sleep

# https://github.com/password123456/setup-selenium-with-chrome-driver-on-ubuntu_debian

def get_shadow_root(driver: WebDriver, element: WebElement) -> WebElement:
    return driver.execute_script('return arguments[0].shadowRoot', element)


def download_render(pattern_uri: str, download_path: str) -> np.ndarray:
    options = webdriver.ChromeOptions()
    prefs = {"download.default_directory": download_path}
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(options=options)

    try:
        driver.get(pattern_uri)
        driver.implicitly_wait(5)

        custom = driver.find_element(by=By.CSS_SELECTOR, value='sampler-test')
        custom = get_shadow_root(driver, custom)
        render_button = custom.find_element(by=By.CSS_SELECTOR, value='#render')
        render_button.click()

        # TODO: This should be a wait condition in a loop
        driver.implicitly_wait(5)
        download_button = custom.find_element(by=By.CSS_SELECTOR, value='#rendered-audio')
        download_button.click()

        # driver.implicitly_wait(10)

        sleep(5)

        # get all audio files in the tmp directory
        download_dir_files = os.listdir(download_path)
        print(download_dir_files)
        all_files = filter(lambda x: os.path.splitext(x)[1] == '.wav', download_dir_files)
        # sort from most to least recently modified
        by_download_date = list(sorted(
            all_files,
            key=lambda x: os.stat(os.path.join(download_path, x)).st_mtime,
            reverse=True))
        print(by_download_date)
        render_file = os.path.join(download_path, by_download_date[0])
        print('RENDER FILE', render_file)

        samples, sr = librosa.load(render_file, sr=22050, mono=True)

        os.remove(render_file)

        return samples

    except Exception as e:
        print(e)
    finally:
        input('Done?')
        driver.quit()


if __name__ == '__main__':
    samples = download_render(
        "https://blog.cochlea.xyz/acc2.html",
        '/home/john/workspace/matching-pursuit')
    plt.plot(samples)
    plt.show()