from conans import ConanFile, CMake, tools
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.scm import Version
from conan.tools.env import VirtualRunEnv, VirtualBuildEnv
from conan.tools.build import check_min_cppstd

import os

required_conan_version = ">=1.53.0"


class ExampleConan(ConanFile):
    name = "example"
    version = "1.0.0"
    license = "<Put the package license here>"
    author = "<Put your name here> <And your email here>"
    url = "<Package recipe repository url here, for issues about the package>"
    description = "<Description of Hello here>"
    topics = ("<Put some tag here>", "<here>", "<and here>")
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "with_test": [True, False],
    }
    default_options = {
        "with_test": False,
    }
    
    def configure(self):
        self.options["libyuv"].with_jpeg = "libjpeg-turbo"

    def requirements(self):
        self.requires("libjpeg-turbo/2.1.5")
        self.requires("libpng/1.6.44")
        self.requires("zlib/1.2.13")
        self.requires("ffmpeg/4.3.2")
        self.requires("opencv/4.5.5@transformer/stable")
        self.requires("libyuv/1768")
        self.requires("cudatoolkit/11.7.1")


    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["CMAKE_POLICY_DEFAULT_CMP0077"] = "NEW"
        if self.options.with_test:
            tc.variables["BUILD_TESTING"] = True
        else:
            tc.variables["BUILD_TESTING"] = False  
        tc.generate()
        tc = CMakeDeps(self)
        tc.generate()
        tc = VirtualRunEnv(self)
        tc.generate()
        tc = VirtualBuildEnv(self)
        tc.generate()
