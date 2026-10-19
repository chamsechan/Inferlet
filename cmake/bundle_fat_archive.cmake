# cmake/bundle_fat_archive.cmake
# Merges multiple static archives into a single fat archive using GNU ar MRI script.
#
# Expected variables:
#   OUTPUT_ARCHIVE      - Path to the final fat archive
#   WRAPPER_ARCHIVE     - Path to the primary library archive
#   DEPENDENCY_ARCHIVES - Pipe-separated list of dependency archives to bundle
#   AR_EXECUTABLE       - Path to ar
#   RANLIB_EXECUTABLE   - Path to ranlib

cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED OUTPUT_ARCHIVE OR NOT DEFINED WRAPPER_ARCHIVE OR NOT DEFINED DEPENDENCY_ARCHIVES)
    message(FATAL_ERROR "bundle_fat_archive.cmake: OUTPUT_ARCHIVE, WRAPPER_ARCHIVE, and DEPENDENCY_ARCHIVES must be specified")
endif()

if(NOT DEFINED AR_EXECUTABLE)
    find_program(AR_EXECUTABLE ar REQUIRED)
endif()

if(NOT DEFINED RANLIB_EXECUTABLE)
    find_program(RANLIB_EXECUTABLE ranlib REQUIRED)
endif()

get_filename_component(OUTPUT_DIR "${OUTPUT_ARCHIVE}" DIRECTORY)
set(BUNDLE_DIR "${OUTPUT_DIR}/fat_bundle")
file(MAKE_DIRECTORY "${BUNDLE_DIR}")
set(FAT_TMP "${BUNDLE_DIR}/libinferlet_fat_tmp.a")
set(MRI_FILE "${BUNDLE_DIR}/fat_bundle.mri")

string(REPLACE "|" ";" DEP_LIST "${DEPENDENCY_ARCHIVES}")

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" OR APPLE)
    find_program(LIBTOOL_EXECUTABLE NAMES libtool REQUIRED)
    message(STATUS "[Inferlet] Combining archives using libtool -static into ${FAT_TMP}...")
    execute_process(
        COMMAND "${LIBTOOL_EXECUTABLE}" -static -o "${FAT_TMP}" "${WRAPPER_ARCHIVE}" ${DEP_LIST}
        RESULT_VARIABLE LIBTOOL_RES
    )
    if(NOT LIBTOOL_RES EQUAL 0)
        message(FATAL_ERROR "bundle_fat_archive: libtool -static failed with exit code ${LIBTOOL_RES}")
    endif()
else()
    # GNU ar's MRI parser does not support paths containing spaces, even quoted.
    # Stage archives under simple relative names and run MRI in that directory.
    configure_file("${WRAPPER_ARCHIVE}" "${BUNDLE_DIR}/wrapper.a" COPYONLY)
    set(MRI_CONTENT "CREATE libinferlet_fat_tmp.a\nADDLIB wrapper.a\n")
    set(DEP_INDEX 0)

    foreach(DEP IN LISTS DEP_LIST)
        if(EXISTS "${DEP}")
            math(EXPR DEP_INDEX "${DEP_INDEX} + 1")
            configure_file("${DEP}" "${BUNDLE_DIR}/dependency_${DEP_INDEX}.a" COPYONLY)
            string(APPEND MRI_CONTENT "ADDLIB dependency_${DEP_INDEX}.a\n")
        else()
            message(FATAL_ERROR "bundle_fat_archive: Dependency archive not found: ${DEP}")
        endif()
    endforeach()

    string(APPEND MRI_CONTENT "SAVE\nEND\n")

    file(WRITE "${MRI_FILE}" "${MRI_CONTENT}")

    message(STATUS "[Inferlet] Combining archives into ${FAT_TMP}...")
    execute_process(
        COMMAND "${AR_EXECUTABLE}" -M
        INPUT_FILE "${MRI_FILE}"
        WORKING_DIRECTORY "${BUNDLE_DIR}"
        RESULT_VARIABLE AR_RES
    )

    if(NOT AR_RES EQUAL 0)
        message(FATAL_ERROR "bundle_fat_archive: ar -M failed with exit code ${AR_RES}")
    endif()
endif()

execute_process(
    COMMAND "${RANLIB_EXECUTABLE}" "${FAT_TMP}"
    RESULT_VARIABLE RANLIB_RES
)

if(NOT RANLIB_RES EQUAL 0)
    message(FATAL_ERROR "bundle_fat_archive: ranlib failed with exit code ${RANLIB_RES}")
endif()

# Replace original archive atomically
file(RENAME "${FAT_TMP}" "${OUTPUT_ARCHIVE}")
file(REMOVE_RECURSE "${BUNDLE_DIR}")
message(STATUS "[Inferlet] Successfully created fat archive: ${OUTPUT_ARCHIVE}")
