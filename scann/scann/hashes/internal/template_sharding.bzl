"""
This module does template substitutions on .tpl.cc files that are necessary to
parallelize the builds of large, heavily-templated methods like LUT16-AVX512.
"""

def _batch_size_sharder_impl(ctx):
    outs = []
    for batch_size in range(1, ctx.attr.max_batch_size + 1):
        out = ctx.actions.declare_file(ctx.label.name + "_{}.cc".format(batch_size))
        ctx.actions.expand_template(
            output = out,
            template = ctx.file.template,
            substitutions = {"{BATCH_SIZE}": str(batch_size)},
        )
        outs.append(out)
    return [DefaultInfo(files = depset(outs))]

batch_size_sharder = rule(
    attrs = {
        "max_batch_size": attr.int(mandatory = True),
        "template": attr.label(
            allow_single_file = [".tpl.cc"],
            mandatory = True,
        ),
    },
    implementation = _batch_size_sharder_impl,
)
