from setuptools import setup, Extension

import pybind11



setup(

    name='video_analyzer_cpp',

    version='1.0',

    description='Fast C++ Video Analyzer',

    ext_modules=[

        Extension(

            'video_analyzer_cpp',

            sources=['frame_analyzer.cpp'],

            include_dirs=[

                pybind11.get_include(),

                '/opt/homebrew/opt/opencv/include/opencv4',

                '/opt/homebrew/include',

                '/usr/local/include'

            ],

            library_dirs=[

                '/opt/homebrew/opt/opencv/lib',

                '/opt/homebrew/lib',

                '/usr/local/lib'

            ],

            libraries=['opencv_core', 'opencv_videoio',

                       'opencv_imgproc', 'opencv_imgcodecs'],

            language='c++',

            extra_compile_args=['-std=c++17', '-O3']

        ),

    ],

)
