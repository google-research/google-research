"""Build definitions for Zebraix."""

def gen_planar_svg(
        name,
        out_file,
        src_file,
        src_dir,
        out_dir,
        extra_args = []):
    """Build an SVG file from a proto definition of planar graph.

    Args:
      name: The name.
      out_file: The output filename.
      src_file: The source filename. Leave empty for sample output.
      src_dir: The source directory.
      out_dir: The output directory.
      extra_args: Extra args.
    """
    if (src_file == ""):
        srcs = []
        located_src_file = "dummy_file"
    else:
        srcs = [src_dir + src_file]
        located_src_file = "$(location " + src_file + ")"
    native.genrule(
        name = name,
        srcs = srcs + [
            ":fontfiles",
        ],
        outs = [
            out_dir + out_file,
        ],
        cmd =

            # Create a temporary FONTS_DIR with OpenSans and set the FONTCONFIG_FILE environment
            # variable to reference a temporary conffile. This fixes the "Fontconfig error: Cannot load default config file"
            # warning.
            "FONTS_DIR=$$(mktemp -d fonts.XXXXXX --tmpdir=\"$${PWD}\");" +
            """
        for FILESET in $(location :fontfiles) ; do
          for FILE in $$(find $$FILESET ! -type d); do
            cp "$$FILE" $$FONTS_DIR/
          done
         done;
         """ + " ".join([
                "  FONTCONFIG_DIR=$$(mktemp -d fontconfig.XXXXXX --tmpdir=\"$${PWD}\")" +
                " && export FONTCONFIG_FILE=$${FONTCONFIG_DIR}/conffile" +
                " && echo \"" +
                "<?xml version=\\\"1.0\\\"?>" +
                "<!DOCTYPE fontconfig SYSTEM \\\"fonts.dtd\\\">" +
                "<fontconfig>" +
                "<dir>$${FONTS_DIR}</dir>" +
                "<cachedir>$${FONTCONFIG_DIR}</cachedir>" +
                "</fontconfig>" +
                "\" > \"$${FONTCONFIG_FILE}\";",
                "$(location //third_party/google_research/google_research/zebraix/tools:zebraix_svg)",
                located_src_file,
                "$(location " + out_dir + out_file + ")",
            ] + extra_args),
        tools = [
            "//third_party/google_research/google_research/zebraix/tools:zebraix_svg",
        ],
        testonly = True,  # Or not - from piffle-0.
    )

def test_planar_svg(
        name,
        out_file,
        goldens_dir,
        out_dir,
        allow_updates):
    """Pass a source file through the lexer, check that it can be reconstructed.

    Use --test_arg="update_dir=$PWD" with local "test strategy" to update
    goldens instead of testing.

    Args:
      name: The name.
      out_file: The output filename.
      goldens_dir: The output directory.
      out_dir: The output directory.
      allow_updates: True/false.
    """
    # src_full_name = out_dir + out_file
    # srcs = [src_full_name]
    # located_src_file = "$(location " + src_full_name + ")"

    further_args = []
    if (allow_updates):
        further_args.append("--allow_updates=true")

    output_full_name = out_dir + out_file  # "$(@D)/" +
    golden_full_name = goldens_dir  # + out_file

    native.sh_test(
        name = name,
        srcs = ["//third_party/google_research/google_research/zebraix/base:diff_test.sh"],
        args = [
            "--golden_file=$(location " + goldens_dir + ":" + out_file + ")",
            "--out_file=$(location " + output_full_name + ")",
        ] + further_args,
        data = [
            golden_full_name + ":" + out_file,
            output_full_name,
        ],
        size = "small",
    )

def gentest_planar_svg(
        name,
        aaa_gen_name,
        aaa_test_name,
        out_file,
        src_file,
        src_dir,
        out_dir,
        goldens_dir,
        extra_args = []):
    """Build an SVG file from a proto definition of planar graph.

    Args:
      name: Macros are required to have a name name.
      aaa_gen_name: The name.
      aaa_test_name: The name.
      out_file: The output filename.
      src_dir: The source directory.
      out_dir: The output directory.
      goldens_dir: The output directory. IGNORED for pass-through tests.
      src_file: The source filename. Leave empty for sample output.
      extra_args: Extra args.
    """

    allow_updates = True

    gen_planar_svg(aaa_gen_name, out_file, src_file, src_dir, out_dir, extra_args)
    test_planar_svg(aaa_test_name, out_file, goldens_dir, out_dir, allow_updates)
