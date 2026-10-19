# cmake/InferletFatArchive.cmake
# Defines a helper to create a fat archive combining a wrapper static library
# with its static dependency archives (llama, ggml, etc.).

function(inferlet_add_fat_archive TARGET_NAME)
    set(DEPENDENCY_TARGETS ${ARGN})

    set(BUNDLE_SCRIPT "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/bundle_fat_archive.cmake")
    set_property(TARGET ${TARGET_NAME} APPEND PROPERTY LINK_DEPENDS "${BUNDLE_SCRIPT}")
    set(WRAPPER_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}_wrapper.a")

    # Build generator expressions for dependency target archive paths
    set(DEP_FILES "")
    foreach(DEP_TARGET ${DEPENDENCY_TARGETS})
        list(APPEND DEP_FILES "$<TARGET_FILE:${DEP_TARGET}>")
    endforeach()

    string(JOIN "|" DEP_FILES_JOINED ${DEP_FILES})

    add_custom_command(
        TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:${TARGET_NAME}>" "${WRAPPER_ARCHIVE}"
        COMMAND ${CMAKE_COMMAND}
            -DOUTPUT_ARCHIVE=$<TARGET_FILE:${TARGET_NAME}>
            -DWRAPPER_ARCHIVE=${WRAPPER_ARCHIVE}
            -DDEPENDENCY_ARCHIVES=${DEP_FILES_JOINED}
            -DAR_EXECUTABLE=${CMAKE_AR}
            -DRANLIB_EXECUTABLE=${CMAKE_RANLIB}
            -DCMAKE_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}
            -P "${BUNDLE_SCRIPT}"
        COMMENT "[Inferlet] Bundling static libraries into fat archive $<TARGET_FILE_NAME:${TARGET_NAME}>"
        VERBATIM
    )
endfunction()
