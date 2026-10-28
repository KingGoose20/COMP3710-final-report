"""Synthetic regression for aggregate replay-boolean export and fail-closed guard."""
from copy import deepcopy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import dataset
from g3 import export_adapter as fix
from g3 import results


def summary():
    return {'study':'G3_Seed42','rows':[{'name':'G3','replay':{'checks':{'attention':{
        'absolute_tolerance':2e-5,'max_absolute_difference':0.,'slice_order_unchanged':True}}}}]}


class ExportTests(unittest.TestCase):
    def test_real_receipt_shape_rejected_before_and_safe_after(self):
        value=summary();original=deepcopy(value)
        with self.assertRaises(dataset.DataError):results.privacy(value)
        public=fix.sanitize_summary(deepcopy(value));results.privacy(public)
        self.assertTrue(public['rows'][0]['replay']['checks']['attention']['ordering_unchanged'])
        self.assertEqual(value,original)

    def test_identifiers_paths_arrays_and_unknown_fields_still_rejected(self):
        for key,value in [('subject_id','synthetic-person'),('scan_id','synthetic-scan'),
                          ('image_path','synthetic/image.jpeg'),('slice_order',['synthetic/image.jpeg']),
                          ('attention_weights',[.5,.5]),('path','C:/Users/private'),('path','C:\\Users\\private'),
                          ('note','PRIVATE_pred.json')]:
            bad=summary();bad['unexpected']={key:value}
            with self.assertRaises(dataset.DataError):results.privacy(fix.sanitize_summary(bad))

    def test_failed_non_boolean_unexpected_or_conflicting_order_checks_rejected(self):
        for value in (False,1,'true',[],{}):
            bad=summary();bad['rows'][0]['replay']['checks']['attention']['slice_order_unchanged']=value
            with self.assertRaises(dataset.DataError):fix.sanitize_summary(bad)
        bad=summary();bad['rows'][0]['name']='unexpected'
        with self.assertRaises(dataset.DataError):fix.sanitize_summary(bad)
        bad=summary();bad['rows'][0]['replay']['checks']['attention']['ordering_unchanged']=True
        with self.assertRaises(dataset.DataError):fix.sanitize_summary(bad)

    def test_scoped_adapter_calls_original_guard_and_restores_it(self):
        guard=results.privacy;seen=[]
        def fake_compare(args):
            value=summary();results.privacy(value);seen.append(value)
        with patch.object(results,'compare',side_effect=fake_compare):fix.compare(None)
        self.assertIs(results.privacy,guard);guard(seen[0])
        bad=summary();bad['patient_predictions']={'subject_id':'synthetic'}
        with patch.object(results,'compare',side_effect=lambda args:results.privacy(bad)):
            with self.assertRaises(dataset.DataError):fix.compare(None)
        self.assertIs(results.privacy,guard)

    def test_check_only_stops_before_export_and_restores_guard(self):
        guard=results.privacy;after=[]
        def fake_compare(args):results.privacy(summary());after.append('would write')
        with patch.object(results,'compare',side_effect=fake_compare):fix.compare(None,True)
        self.assertEqual(after,[]);self.assertIs(results.privacy,guard)

    def test_repaired_summary_passes_frozen_bundle_guard_allowlist_roundtrip(self):
        import json,zipfile
        from types import SimpleNamespace
        from g3 import storage as st
        with tempfile.TemporaryDirectory(prefix='g3 export repair spaces ') as tmp:
            root=Path(tmp);folder=root/'aggregate_exports'/('a'*20);folder.mkdir(parents=True)
            value=fix.sanitize_summary(summary());st.immutable_json(folder/'comparison.json',value)
            (folder/'PRIVATE_extra.pt').write_bytes(b'synthetic excluded')
            st.immutable_json(root/'latest_aggregate.json',{'directory':folder.name})
            args=SimpleNamespace(study_root=root)
            with patch.object(st,'context',return_value=({},root,{},None,{})):
                results.bundle(args);results.bundle(args)
            with zipfile.ZipFile(folder/'COMP3710_G3_Seed42_Results.zip') as z:
                self.assertEqual(z.namelist(),['comparison.json']);self.assertEqual(json.loads(z.read('comparison.json')),value)


if __name__=='__main__':unittest.main(verbosity=2)
