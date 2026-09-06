// Run after a Web build: node web/test-espeak-data.mjs build-web
import assert from 'node:assert/strict';
import { mkdtempSync, readdirSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const build = resolve(process.argv[2] || 'build-web');
const temporary = mkdtempSync(join(tmpdir(), 'kokopop-espeak-'));
const destination = join(temporary, 'data');
const select = (languages) => spawnSync('cmake', [
  `-DDATA_SOURCE=${build}/espeak-native/espeak-ng-data`,
  `-DDICT_SOURCE=${build}/_deps/espeak-src/dictsource`,
  `-DDATA_DEST=${destination}`,
  `-DLANGUAGES=${languages}`,
  '-P', fileURLToPath(new URL('./select-espeak-data.cmake', import.meta.url)),
], { encoding: 'utf8' });

try {
  // Reuse the destination to catch stale dictionaries after narrowing a build.
  for (const [languages, expected] of [
    ['all', null], ['en;fr', ['en_dict', 'fr_dict']], ['en', ['en_dict']],
    ['ka', ['en_dict', 'ka_dict', 'ru_dict']], // Georgian switches to Russian.
  ]) {
    const result = select(languages);
    assert.equal(result.status, 0, result.stderr);
    const dictionaries = readdirSync(destination).filter((name) => name.endsWith('_dict')).sort();
    if (expected) assert.deepEqual(dictionaries, expected);
    else assert.ok(dictionaries.length > 100);
    for (const file of ['phondata', 'phonindex', 'phontab', 'intonations', 'lang/gmw/en']) {
      assert.ok(existsSync(join(destination, file)), file);
    }
    assert.equal(existsSync(join(destination, 'lang/roa/fr')), languages === 'all' || languages === 'en;fr');
  }
  for (const invalid of ['', 'en-us', 'nonexistent', 'all;en']) {
    const result = select(invalid);
    assert.notEqual(result.status, 0);
    assert.match(result.stderr, /KOKOPOP_WEB_LANGUAGES/);
  }
  console.log('eSpeak selection tests passed');
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
