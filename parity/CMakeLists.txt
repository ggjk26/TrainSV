cmake_minimum_required(VERSION 3.25)
project(TrainSVParity VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(JUCE
    URL https://github.com/juce-framework/JUCE/archive/be29c81492b6151c8ea8d14c840e1311963b3a83.tar.gz
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(JUCE)

file(GLOB ENGINE_SOURCES CONFIGURE_DEPENDS engine/synthesis/*.cpp)
add_library(ReferenceEngine STATIC ${ENGINE_SOURCES})
target_include_directories(ReferenceEngine PUBLIC engine)
target_compile_definitions(ReferenceEngine PUBLIC JUCE_WEB_BROWSER=0 JUCE_USE_CURL=0)
target_link_libraries(ReferenceEngine PRIVATE
    juce::juce_core
    juce::juce_audio_basics
    juce::juce_dsp
    juce::juce_recommended_config_flags
)

juce_add_console_app(OpenSVParity
    PRODUCT_NAME "opensv-parity"
    COMPANY_NAME "TrainSV"
    BUNDLE_ID "org.trainsv.parity"
)
set_target_properties(OpenSVParity PROPERTIES OUTPUT_NAME "opensv-parity")
target_sources(OpenSVParity PRIVATE main.cpp)
target_link_libraries(OpenSVParity PRIVATE
    ReferenceEngine
    juce::juce_core
    juce::juce_audio_basics
    juce::juce_dsp
    juce::juce_recommended_config_flags
    juce::juce_recommended_warning_flags
)
if(MSVC)
    target_compile_options(ReferenceEngine PRIVATE /utf-8 /permissive- /EHsc)
    target_compile_options(OpenSVParity PRIVATE /utf-8 /permissive- /EHsc)
endif()
