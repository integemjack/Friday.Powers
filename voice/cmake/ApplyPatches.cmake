# 给 FetchContent 拉下来的第三方源码打自己的补丁：配置时（FetchContent_Populate 之后）调
#   voice_apply_patches(<源码目录> <补丁目录>)
# 补丁目录里的 *.patch 按文件名顺序用 git apply 打；已经打过的（反着能打上）跳过，所以每次配置都调也没事，
# FETCHCONTENT_SOURCE_DIR_<名字> 指到别处的源码（本机离线编）也照样打上。
# 源码目录一般在本仓库的构建目录里：GIT_CEILING_DIRECTORIES 拦着 git 往上找仓库，不然 git apply 会按外面仓库的根算路径、
# 把补丁当成「不在这个目录里」静默跳过。补丁文件要是 LF（patches/.gitattributes 里 -text，Windows 上检出不转 CRLF）
find_package(Git QUIET)

function(voice_apply_patches src_dir patch_dir)
    file(GLOB patches "${patch_dir}/*.patch")
    list(SORT patches)
    if(NOT patches)
        return()
    endif()
    if(NOT GIT_FOUND)
        message(FATAL_ERROR "要 git 才能给 ${src_dir} 打补丁（${patch_dir}）")
    endif()
    get_filename_component(parent "${src_dir}" DIRECTORY)
    set(git ${CMAKE_COMMAND} -E env "GIT_CEILING_DIRECTORIES=${parent}" "${GIT_EXECUTABLE}")
    foreach(patch IN LISTS patches)
        get_filename_component(name "${patch}" NAME)
        execute_process(COMMAND ${git} apply --reverse --check --whitespace=nowarn "${patch}"
            WORKING_DIRECTORY "${src_dir}" RESULT_VARIABLE applied OUTPUT_QUIET ERROR_QUIET)
        if(applied EQUAL 0)
            continue()
        endif()
        execute_process(COMMAND ${git} apply --whitespace=nowarn "${patch}"
            WORKING_DIRECTORY "${src_dir}" RESULT_VARIABLE result ERROR_VARIABLE err)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "补丁 ${name} 打不上 ${src_dir}：${err}\n"
                "（源码被改过、或者打过旧版的补丁：删掉这个源码目录重新配置）")
        endif()
        message(STATUS "已打补丁：${name} → ${src_dir}")
    endforeach()
endfunction()
