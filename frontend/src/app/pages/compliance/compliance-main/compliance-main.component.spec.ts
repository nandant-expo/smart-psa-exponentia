import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ComplianceMainComponent } from './compliance-main.component';

describe('ComplianceMainComponent', () => {
  let component: ComplianceMainComponent;
  let fixture: ComponentFixture<ComplianceMainComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ComplianceMainComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ComplianceMainComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
