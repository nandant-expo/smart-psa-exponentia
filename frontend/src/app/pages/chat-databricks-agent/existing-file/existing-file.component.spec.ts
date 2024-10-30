import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExistingFileComponent } from './existing-file.component';
import { NgbActiveModal } from '@ng-bootstrap/ng-bootstrap';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { TableModule } from 'primeng/table';
import { ProgressBarModule } from 'primeng/progressbar';

describe('ExistingFileComponent', () => {
  let component: ExistingFileComponent;
  let fixture: ComponentFixture<ExistingFileComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ExistingFileComponent ],
      providers: [ NgbActiveModal ],
      imports: [ HttpClientTestingModule, TableModule, ProgressBarModule ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ExistingFileComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
