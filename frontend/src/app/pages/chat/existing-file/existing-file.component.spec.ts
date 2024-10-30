import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExistingFileComponent } from './existing-file.component';
import { ToastModule } from 'primeng/toast';
import { TableModule } from 'primeng/table';
import { ProgressBarModule } from 'primeng/progressbar';
import { FormsModule, ReactiveFormsModule } from '@angular/forms';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { NgbActiveModal } from '@ng-bootstrap/ng-bootstrap';

describe('ExistingFileComponent', () => {
  let component: ExistingFileComponent;
  let fixture: ComponentFixture<ExistingFileComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [
        HttpClientTestingModule,
        ToastModule,
        TableModule,
        ProgressBarModule,
        FormsModule,
        ReactiveFormsModule
      ],
      declarations: [ ExistingFileComponent ],
      providers: [ NgbActiveModal ]      
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
