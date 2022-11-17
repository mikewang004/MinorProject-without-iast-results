      Program Testiast
      Implicit None

      Include 'commons.inc'

      Integer J,I,Nterm_max,Carrier_gas
      
      Double Precision P,Yi(Maxcomp),Xi1(Maxcomp)
     $     ,Ni(Maxcomp),Molfrac(Maxcomp,Maxterm)

C     An Inert Component Should Have
C     Nterm=1
C     Langmuir=True
C     Ki=0
C     Nimax Not Equal To Zero

      
      Ncomp = 3
      Nterm_max = 2

C     He
      Ki(1,1)       = 0.0d0
      Ki(1,2)       = 0.0d0
      Nimax(1,1)    = 0.1d0
      Nimax(1,2)    = 0.1d0
      Pow(1,1)      = 1.0d0
      Pow(1,2)      = 1.0d0
      Langmuir(1,1) = .True.
      Langmuir(1,2) = .True.      

C     C4    
      Ki(2,1)       = 2.26d-04
      Ki(2,2)       = 2.35d-05
      Nimax(2,1)    = 7.000d-01
      Nimax(2,2)    = 1.024d+00
      Pow(2,1)      = 1.0d0
      Pow(2,2)      = 1.0d0
      Langmuir(2,1) = .True.
      Langmuir(2,2) = .True.
      
C     2mC3
      Ki(3,1)       = 9.86d-05
      Ki(3,2)       = 1.00d-07
      Nimax(3,1)    = 7.000d-01
      Nimax(3,2)    = 9.000d-01
      Pow(3,1)      = 1.0d0
      Pow(3,2)      = 1.0d0
      Langmuir(3,1) = .True.
      Langmuir(3,2) = .True.

      Carrier_gas = 1       
      P   = 1.0d4

      Yi(1) = 0.00d0
      Yi(2) = 0.50d0      
      Yi(3) = 0.50d0      


      Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)

      write(6,*) 'Ni(2)   ','Ni(3)   '
      write(6,'(2e20.10)') Ni(2),Ni(3)

C      stop

      Do J = 0, 8
         If(J.Lt.8) Then
            Do I = 1, 3, 2
               P = Dble(I*10**J)
               Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
               Write(25,'(5e20.10)') P,Ni(2),Ni(3)
            Enddo
         Else
            P = Dble(1.0*10**J)
            Call Seg_Iast(Ni,Xi1,Molfrac,Yi,P,Nterm_max,Carrier_gas)
            Write(25,'(5e20.10)') P,Ni(2),Ni(3)
         Endif
      Enddo
      
      
      Stop
      End
      
