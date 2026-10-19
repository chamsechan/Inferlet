# tests/abi/test_symbols.cmake
# Usage: cmake -DARCHIVE_FILE=<path> -DEXPECTED_SYMBOLS=<path> -P test_symbols.cmake

if (NOT DEFINED ARCHIVE_FILE OR NOT DEFINED EXPECTED_SYMBOLS)
    message(FATAL_ERROR "ARCHIVE_FILE and EXPECTED_SYMBOLS must be defined")
endif()

find_program(NM_TOOL nm REQUIRED)

execute_process(
    COMMAND ${NM_TOOL} -g -U ${ARCHIVE_FILE}
    OUTPUT_VARIABLE NM_OUTPUT
    RESULT_VARIABLE NM_RESULT
)

if (NOT NM_RESULT EQUAL 0)
    message(FATAL_ERROR "nm failed on ${ARCHIVE_FILE}")
endif()

string(REGEX MATCHALL " [A-Z] +_?(inferlet_[A-Za-z0-9_]+)" MATCHED_LINES "${NM_OUTPUT}")
set(FOUND_SYMBOLS "")
foreach(LINE ${MATCHED_LINES})
    string(REGEX REPLACE " [A-Z] +_?(inferlet_[A-Za-z0-9_]+)" "\\1" SYM "${LINE}")
    list(APPEND FOUND_SYMBOLS "${SYM}")
endforeach()
list(REMOVE_DUPLICATES FOUND_SYMBOLS)
list(SORT FOUND_SYMBOLS)

file(STRINGS ${EXPECTED_SYMBOLS} EXPECTED_LIST)
list(SORT EXPECTED_LIST)

if (NOT FOUND_SYMBOLS STREQUAL EXPECTED_LIST)
    message(STATUS "Expected symbols: ${EXPECTED_LIST}")
    message(STATUS "Found symbols:    ${FOUND_SYMBOLS}")
    message(FATAL_ERROR "Symbol comparison failed! Defined exported symbols do not match expected list.")
else()
    message(STATUS "Symbol check passed: all 64 exported Inferlet symbols matched exactly.")
endif()
