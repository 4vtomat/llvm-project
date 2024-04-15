set(LLVM_EXPORTED_SYMBOL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/gold.exports)

if( LLVM_ENABLE_PIC AND LLVM_BINUTILS_INCDIR )
  include_directories( ${LLVM_BINUTILS_INCDIR} )

  set(LLVM_LINK_COMPONENTS
     ${LLVM_TARGETS_TO_BUILD}
     Linker
     LTO
     BitWriter
     IPO
     TargetParser
     )

<<<<<<< HEAD
# SIFIVE_CUSTOMIZATON
  add_llvm_library(LLVMgold MODULE INSTALL_WITH_TOOLCHAIN
# end SIFIVE_CUSTOMIZATION	  
=======
  add_llvm_library(LLVMgold MODULE INSTALL_WITH_TOOLCHAIN
>>>>>>> 6f1e23b47d428d792866993ed26f4173d479d43d
    gold-plugin.cpp
    )

endif()
