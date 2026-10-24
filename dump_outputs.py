"""Print text outputs of the executed notebook (for review); flags errors/tracebacks/stderr, image sizes."""
import base64
import sys

import nbformat

nb = nbformat.read(sys.argv[1] if len(sys.argv) > 1 else 'dz_topic_6_Джафарлі_Фарід.ipynb', as_version=4)
n_err = n_img = img_bytes = 0
for i, c in enumerate(nb.cells):
    if c.cell_type != 'code':
        continue
    print(f'\n===== cell {i} =====')
    print(c.source.splitlines()[0][:100])
    print('-----')
    for o in c.get('outputs', []):
        if o.output_type == 'error':
            n_err += 1
            print('ERROR:', o.ename, o.evalue)
        elif o.output_type == 'stream':
            print(('[stderr] ' if o.name == 'stderr' else '') + o.text[:4000])
        else:
            data = o.get('data', {})
            fmt = next((f for f in ('image/png', 'image/jpeg') if f in data), None)
            if fmt:
                n_img += 1
                size = len(base64.b64decode(data[fmt]))
                img_bytes += size
                print(f'[{fmt} {size / 1e6:.2f} MB]')
            elif 'text/plain' in data:
                print(data['text/plain'][:4000])
print(f'\nerrors: {n_err}, images: {n_img}, image bytes: {img_bytes / 1e6:.1f} MB')
