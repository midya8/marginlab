'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),path=require('node:path'),{check}=require('../docs/certificate-verify.js');
for(const id of ['linear-0.01','linear-0.1','linear-1','linear-10'])require('../docs/certificate-data-'+id+'.js');
test('independent BigInt verifier checks all four exact 341x30 full-data proofs',()=>{
 for(const [id,p] of Object.entries(globalThis.MARGIN_CERTIFICATES)){const result=check(p);assert.equal(result.valid,true);assert.equal(result.rows,341);assert.equal(result.features,30);assert.equal(result.exact_gap,p.objectives.gap);assert.ok(result.repaired_rows>0);}
});
test('independent BigInt verifier rejects exact constraint/objective/repair/schema tampering',()=>{
 const original=globalThis.MARGIN_CERTIFICATES['linear-1'];
 for(const mutate of [p=>p.dual.alpha[0]='1/2',p=>p.primal.w[0]='0/1',p=>p.primal.slacks[0]='-1/1',p=>p.objectives.gap='0/1',p=>p.repair.original_equality='0/1',p=>p.repair.l1_change='0/1',p=>p.problem.x[p.repair.original_alpha.findIndex(a=>a!=='0/1')][0]=42,p=>p.problem.C=.25,p=>p.version=true,p=>p.dual.alpha[0]='02/4']){const p=structuredClone(original);mutate(p);assert.throws(()=>check(p));}
});

test('a valid changed declared problem is not original-data authentication',()=>{const p=structuredClone(MARGIN_CERTIFICATES['linear-1']);assert.equal(p.repair.original_alpha[0],'0/1');p.problem.x[0][0]=42;assert.equal(check(p).valid,true);});
