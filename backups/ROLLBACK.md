# OPSD original backup / rollback

Do not push to `origin` (upstream paper repo). All backup refs are local.

## Restore original trainer (discard OG-OPSD edits)

```bash
cd /home/aochang/kongjing/OPSD
git checkout backup/original-opsd -- opsd_trainer.py opsd_train.py data_collator.py
# or copy files:
cp backups/original-ae7d251/opsd_trainer.py .
cp backups/original-ae7d251/opsd_train.py .
cp backups/original-ae7d251/data_collator.py .
```

## Git refs

- Branch `backup/original-opsd` = upstream `ae7d251`
- Tag `original-opsd-repro` = same commit
- Branch `og-opsd` = OG-OPSD work
- File snapshot: `backups/original-ae7d251/`

Local reproduction artifacts (`runs/repro_100_gpu0`, `eval_results/`) are not in git
and must not be deleted.
