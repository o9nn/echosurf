{
  "targets": [
    {
      "target_name": "echo_ml",
      "sources": [
        "src/echo_tensor.c",
        "src/echo_reservoir.c",
        "src/echo_layers.c",
        "src/echo_noi_bridge.cpp"
      ],
      "include_dirs": [
        "include"
      ],
      "cflags": [
        "-O3",
        "-march=native",
        "-ffast-math",
        "-Wall"
      ],
      "cflags_c": [
        "-std=c11"
      ],
      "cflags_cc": [
        "-std=c++17"
      ],
      "conditions": [
        ["OS=='linux'", {
          "cflags": [
            "-mavx2",
            "-mfma"
          ]
        }],
        ["OS=='mac'", {
          "xcode_settings": {
            "OTHER_CFLAGS": [
              "-O3",
              "-march=native",
              "-ffast-math"
            ],
            "OTHER_CPLUSPLUSFLAGS": [
              "-std=c++17"
            ]
          }
        }],
        ["OS=='win'", {
          "msvs_settings": {
            "VCCLCompilerTool": {
              "Optimization": 3,
              "EnableEnhancedInstructionSet": 5
            }
          }
        }]
      ]
    }
  ]
}
