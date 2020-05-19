PROJECT(latex_project NONE)

cmake_minimum_required(VERSION 2.8)

SET(LATEX_OUTPUT_PATH build)
INCLUDE(UseLATEX.cmake)

file(GLOB_RECURSE contents_files RELATIVE ${CMAKE_SOURCE_DIR} contents/*.tex)

#Copy beamer theme
file(COPY  ${CMAKE_SOURCE_DIR}/beamer_theme/ DESTINATION ./ )


ADD_LATEX_DOCUMENT(
    example.tex
    #INPUTS ${contents_files}
    #IMAGE_DIRS ./
    BIBFILES lib.bib
    #USE_INDEX
    #USE_GLOSSARY
    #FILTER_OUTPUT
    )
