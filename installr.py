import subprocess
import platform


def check_r_installed():
    current_platform = platform.system()

    if current_platform == "Windows":
        # Check if R is installed on Windows by checking the registry
        try:
            subprocess.run(
                ["reg", "query", "HKLM\\Software\\R-core\\R"], check=True
            )
            print("R is already installed on Windows.")
            return True
        except subprocess.CalledProcessError:
            print("R is not installed on Windows.")
            return False

    elif current_platform == "Linux":
        # Check if R is installed on Linux by checking if the 'R' executable is available
        try:
            subprocess.run(["which", "R"], check=True)
            print("R is already installed on Linux.")
            return True
        except subprocess.CalledProcessError:
            print("R is not installed on Linux.")
            return False

    elif current_platform == "Darwin":  # macOS
        # Check if R is installed on macOS by checking if the 'R' executable is available
        try:
            subprocess.run(["which", "R"], check=True)
            print("R is already installed on macOS.")
            return True
        except subprocess.CalledProcessError:
            print("R is not installed on macOS.")
            return False

    else:
        print("Unsupported platform. Unable to check for R installation.")
        return False


def install_r():
    current_platform = platform.system()

    if current_platform == "Windows":
        # Install R on Windows using PowerShell
        install_command = "Start-Process powershell -Verb runAs -ArgumentList '-Command \"& {Invoke-WebRequest https://cran.r-project.org/bin/windows/base/R-4.1.2-win.exe -OutFile R.exe}; Start-Process R.exe -ArgumentList '/SILENT' -Wait}'"
        subprocess.run(install_command, shell=True)

    elif current_platform == "Linux":
        # Install R on Linux using the appropriate package manager (e.g., apt-get)
        install_command = (
            "sudo apt update -qq && sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9"
            + "&& sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu focal-cran40/'"
            + "&& sudo apt update"
            + "&& sudo apt install r-base"
        )
        subprocess.run(install_command, shell=True)

    elif current_platform == "Darwin":  # macOS
        # Install R on macOS using Homebrew
        install_command = "brew install r"
        subprocess.run(install_command, shell=True)

    else:
        print("Unsupported platform. Unable to install R.")